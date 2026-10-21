// DeepSeek PoW 助手 —— 加载 DeepSeek 官方 WASM 求解，供 Python 通过子进程调用。
// 用法:  node pow_helper.cjs < solve_input.json
//   输入(JSON): { algorithm, challenge, salt, difficulty, expire_at, signature, target_path }
//   输出(stdout): base64 编码的 X-Ds-Pow-Response 头值
//
// 为什么不自己写哈希: DeepSeekHashV1 是非标准 23 轮 Keccak-256，纯 Python 重写
// 慢且易错。直接跑 DeepSeek 官方 wasm(浏览器同款)保证逐字节一致。wasmtime 装不上，
// 故用 Node 原生 WebAssembly 引擎驱动。
'use strict';
const fs = require('fs');
const path = require('path');

const WASM_PATH = path.join(__dirname, 'providers', 'deepseek_pow.wasm');

function loadWasm() {
  const buf = fs.readFileSync(WASM_PATH);
  const mod = new WebAssembly.Module(buf);
  const inst = new WebAssembly.Instance(mod, { wbg: {} });
  return inst.exports;
}

const ex = loadWasm();
const enc = new TextEncoder();
let _mem = null;
function u8() {
  if (!_mem || _mem.byteLength === 0) _mem = new Uint8Array(ex.memory.buffer);
  return _mem;
}
// 写字符串进 wasm 内存，返回 [ptr, len]（对应 __wbindgen_export_0 = malloc）
function writeStr(s) {
  const b = enc.encode(s);
  const ptr = ex.__wbindgen_export_0(b.length, 1) >>> 0;
  u8().subarray(ptr, ptr + b.length).set(b);
  return [ptr, b.length];
}

// 复刻原版 calculateHash: wasm_solve(challenge, prefix, difficulty) -> nonce
function solve(challenge, prefix, difficulty) {
  const retptr = ex.__wbindgen_add_to_stack_pointer(-16);
  const [cp, cl] = writeStr(challenge);
  const [pp, pl] = writeStr(prefix);
  ex.wasm_solve(retptr, cp, cl, pp, pl, Number(difficulty));
  const dv = new DataView(ex.memory.buffer);
  const status = dv.getInt32(retptr + 0, true);   // 0 = 未找到
  const nonce = dv.getFloat64(retptr + 8, true);
  ex.__wbindgen_add_to_stack_pointer(16);
  if (status === 0) return null;
  return nonce;
}

// 导出单哈希(keccak23)，供离线自检构造可解挑战
function hash1(str) {
  const retptr = ex.__wbindgen_add_to_stack_pointer(-16);
  const [p, l] = writeStr(str);
  ex.wasm_deepseek_hash_v1(retptr, p, l);
  const ptr = new DataView(ex.memory.buffer).getInt32(retptr + 0, true);
  ex.__wbindgen_add_to_stack_pointer(16);
  const mem = u8();
  let s = '';
  for (let i = 0; i < 200; i++) {
    const c = mem[ptr + i];
    if (c === 0) break;
    s += String.fromCharCode(c);
  }
  return s;
}

// ---- 主流程: 从 stdin 读 challenge，求解，输出 base64 头 ----
function main() {
  const input = fs.readFileSync(0, 'utf8').trim();
  const c = JSON.parse(input);

  // 自检模式: {__selftest: true, ...}
  if (c.__selftest) {
    const prefix = `${c.salt}_${c.expire_at}_`;
    const full = `${prefix}${c.nonce}`;
    const h = hash1(full);                       // 期望 = keccak23(prefix+nonce)
    const found = solve(h, prefix, c.difficulty); // 用该期望反解
    process.stdout.write(JSON.stringify({
      prefix, computedHash: h, solvedNonce: found,
      ok: found === Number(c.nonce),
    }));
    return;
  }

  if (c.algorithm !== 'DeepSeekHashV1') {
    process.stderr.write('unsupported algorithm: ' + c.algorithm);
    process.exit(2);
  }
  const prefix = `${c.salt}_${c.expire_at}_`;
  const answer = solve(c.challenge, prefix, c.difficulty);
  if (answer === null) {
    process.stderr.write('pow solve failed: no nonce found within difficulty');
    process.exit(3);
  }
  const payload = {
    algorithm: c.algorithm,
    challenge: c.challenge,
    salt: c.salt,
    answer: answer,
    signature: c.signature || '',
    target_path: c.target_path || '/api/v0/chat/completion',
  };
  process.stdout.write(Buffer.from(JSON.stringify(payload)).toString('base64'));
}

main();
