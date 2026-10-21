"""
DeepSeek PoW 求解 —— 用 wasmtime 直接驱动 DeepSeek 官方 WASM。

背景
----
DeepSeek /api/v0/chat/completion 要求每个请求带 X-Ds-Pow-Response 头。
算法 DeepSeekHashV1 = 非标准 23 轮 Keccak-256：
    找 nonce ∈ [0, difficulty)，使 keccak23(f"{salt}_{expire_at}_{nonce}") == challenge

为什么用 wasmtime 跑官方 WASM 而不是纯 Python 重写：
    23 轮 Keccak 是非标准变体，纯 Python 实现慢(实测同难度 66s vs WASM 0.3s)且易错。
    直接跑 DeepSeek 官方 wasm(浏览器同款)保证逐字节一致。

    曾用 Node 子进程驱动（pow_helper.cjs），现改用 wasmtime（纯 Python 运行时），
    消除 Node 依赖，便于 PyInstaller 打包成单文件 exe。

WASM 路径兼容两种运行方式：
  - 源码运行:  providers/deepseek_pow.wasm（项目内）
  - PyInstaller onefile: 打包时经 --add-data 放进 sys._MEIPASS
"""

import json
import base64
import os
import struct
import sys

import wasmtime

# ---- 定位 WASM 文件（兼容 PyInstaller 打包） ----
def _wasm_path() -> str:
    # PyInstaller onefile: 资源在 sys._MEIPASS 下
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "providers", "deepseek_pow.wasm")


class _DeepSeekPow:
    """单例持有 wasmtime 实例，复用 Store（一次加载，多次求解）。"""

    def __init__(self):
        engine = wasmtime.Engine()
        self.store = wasmtime.Store(engine)
        module = wasmtime.Module(self.store.engine, open(_wasm_path(), "rb").read())
        self.instance = wasmtime.Instance(self.store, module, [])
        self.mem = self.instance.exports(self.store)["memory"]

    def _write_str(self, s: str):
        b = s.encode("utf-8")
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, len(b), 1)
        self.mem.write(self.store, b, ptr)
        return ptr, len(b)

    def _hash1(self, s: str) -> str:
        """单次 keccak23，返回 hex 字符串（对应 wasm_deepseek_hash_v1）。"""
        ex = self.instance.exports(self.store)
        retptr = ex["__wbindgen_add_to_stack_pointer"](self.store, -16)
        p, l = self._write_str(s)
        ex["wasm_deepseek_hash_v1"](self.store, retptr, p, l)
        mv = bytes(self.mem.read(self.store))
        ptr = int.from_bytes(mv[retptr:retptr + 4], "little", signed=True)
        ex["__wbindgen_add_to_stack_pointer"](self.store, 16)
        return bytes(mv[ptr:ptr + 200]).split(b"\x00")[0].decode("ascii", "replace")

    def solve(self, challenge: str, prefix: str, difficulty: int) -> int:
        """wasm_solve: 返回满足 keccak23(prefix+nonce)==challenge 的 nonce。"""
        ex = self.instance.exports(self.store)
        retptr = ex["__wbindgen_add_to_stack_pointer"](self.store, -16)
        cp, cl = self._write_str(challenge)
        pp, pl = self._write_str(prefix)
        ex["wasm_solve"](self.store, retptr, cp, cl, pp, pl, float(difficulty))
        mv = bytes(self.mem.read(self.store))
        status = int.from_bytes(mv[retptr:retptr + 4], "little", signed=True)
        nonce = struct.unpack_from("<d", mv, retptr + 8)[0]
        ex["__wbindgen_add_to_stack_pointer"](self.store, 16)
        if status == 0:
            raise RuntimeError(
                f"PoW 求解失败：未在 difficulty={difficulty} 内找到 nonce。"
            )
        return int(nonce)


_solver = None


def _get_solver() -> _DeepSeekPow:
    global _solver
    if _solver is None:
        _solver = _DeepSeekPow()
    return _solver


def build_pow_header(challenge_obj: dict, target_path: str = "/api/v0/chat/completion") -> str:
    """
    入参 challenge_obj 来自 DeepSeek 返回的 biz_data.challenge：
      { algorithm, challenge, salt, difficulty, expire_at, signature }
    返回 X-Ds-Pow-Response 头的值：base64( JSON )。
    """
    if challenge_obj.get("algorithm") != "DeepSeekHashV1":
        raise ValueError(f"不支持的 PoW 算法: {challenge_obj.get('algorithm')}")

    prefix = f"{challenge_obj['salt']}_{int(challenge_obj['expire_at'])}_"
    answer = _get_solver().solve(
        challenge_obj["challenge"],
        prefix,
        int(challenge_obj["difficulty"]),
    )

    payload = {
        "algorithm": challenge_obj["algorithm"],
        "challenge": challenge_obj["challenge"],
        "salt": challenge_obj["salt"],
        "answer": answer,
        "signature": challenge_obj.get("signature", ""),
        "target_path": target_path,
    }
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


if __name__ == "__main__":
    # 离线自检: 自构造可解挑战，验证 wasmtime 求解链路。
    salt, expire_at, nonce = "testsalt", 1700000000, 12345
    prefix = f"{salt}_{expire_at}_"
    h = _get_solver()._hash1(f"{prefix}{nonce}")
    solved = _get_solver().solve(h, prefix, 200000)
    ok = solved == nonce
    print(f"self-check: nonce={solved} expected={nonce} ok={ok}")
    sys.exit(0 if ok else 1)
