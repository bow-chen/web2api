"""
web2api 服务器 —— 暴露 OpenAI 兼容接口，内部转发到 DeepSeek 网页版。

对外接口：
  GET  /health               健康检查
  GET  /v1/models            模型列表（OpenAI 格式）
  POST /v1/chat/completions  对话补全，支持 stream=true/false

安全默认：
  - 默认只绑 127.0.0.1
  - config.yaml 里若设了 server.api_key，则要求 Authorization: Bearer <key>
"""

import json
import time
import uuid
import yaml

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek import DeepSeekAdapter
from sse import iter_text_deltas_async, sse_frame
from throttle import UpstreamThrottle

# ---------- 加载配置 ----------
# 查找 config.yaml 的位置，按优先级：
#   1) 环境变量 AFS_CONFIG 指定的路径
#   2) exe/脚本所在目录旁的 config.yaml（打包后用户可改）
#   3) 当前工作目录的 config.yaml（源码运行）
#   4) 打包进 exe 的默认配置（仅当上面都不存在，正常不会走到）
import os as _os
import sys as _sys

def _find_config():
    env = _os.environ.get("AFS_CONFIG")
    if env:
        return env, "env"
    # 1) exe 所在目录旁的 config.yaml（打包后用户把 config 放 exe 旁边即可）
    #    PyInstaller onefile 下 __file__ 在 _MEIPASS 解压目录，须用 sys.executable 定位 exe。
    exe_dir = _os.path.dirname(_os.path.abspath(_sys.executable))
    meipass = getattr(_sys, "_MEIPASS", None)
    candidates = [_os.path.join(exe_dir, "config.yaml")]
    # 2) 源码运行时：脚本所在目录。（打包运行时 __file__ 指向只读的 _MEIPASS，跳过）
    src_dir = _os.path.dirname(_os.path.abspath(__file__))
    if meipass is None:
        candidates.append(_os.path.join(src_dir, "config.yaml"))
    # 3) 当前工作目录
    candidates.append(_os.path.join(_os.getcwd(), "config.yaml"))
    for cand in candidates:
        if _os.path.exists(cand):
            return cand, "user"
    # 4) 没有任何用户配置：返回应生成 config.yaml 的目录，由 main 生成。
    #    打包运行时生成到 exe 旁；源码运行时生成到脚本（项目）目录。
    gen_dir = src_dir if meipass is None else exe_dir
    return _os.path.join(gen_dir, "config.yaml"), "missing"

CONFIG_PATH, CONFIG_KIND = _find_config()

if CONFIG_KIND == "missing":
    # 没有用户配置：用默认值启动，__main__ 会生成 config.yaml 提示填写。
    # 这样 import 阶段不碰任何不存在的文件 / 只读的 _MEIPASS。
    CFG = {
        "server": {"host": "127.0.0.1", "port": 8080, "api_key": "",
                   "min_request_interval_s": 12},
        "deepseek": {"token": "PASTE_YOUR_DEEPSEEK_USER_TOKEN_HERE",
                     "models": ["DeepSeek-V4.1-Flash", "deepseek-v4-pro"]},
    }
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

SERVER = CFG.get("server", {})
HOST = SERVER.get("host", "127.0.0.1")
PORT = int(SERVER.get("port", 8080))
API_KEY = SERVER.get("api_key", "") or ""

DS_TOKEN = CFG["deepseek"]["token"]
MODELS = CFG["deepseek"].get("models", ["DeepSeek-V4.1-Flash"])

# 风控防护：上游最小请求间隔（秒）。实测 DeepSeek 单分钟 >10 次触发账号级禁言，
# 这里默认 12s ≈ 5 次/分钟，远低于禁言线。串行由 UpstreamThrottle 内部保证。
MIN_INTERVAL_S = float(SERVER.get("min_request_interval_s", 12))

app = FastAPI(title="web2api")

# 复用一个 httpx 连接池（HTTP/2 由 DeepSeek 侧决定，这里走 HTTP/1.1 足够）
_http = httpx.AsyncClient(http2=False, follow_redirects=True)

# 全局节流器：串行化 + 最小间隔，避免并发/高频触发 DeepSeek 网页端风控
_throttle = UpstreamThrottle(MIN_INTERVAL_S)


def _err(status: int, message: str, err_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


def _check_auth(request: Request):
    """若配置了 api_key，校验 Bearer。未配置则放行（本机自用）。"""
    if not API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {API_KEY}":
        return True
    return False


def _new_chunk(model: str, delta: dict, finish=None, cid=None):
    return {
        "id": cid,
        "model": model,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "created": int(time.time()),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "deepseek"}


@app.get("/v1/models")
async def list_models(request: Request):
    if not _check_auth(request):
        return _err(401, "Invalid API key", "authentication_error")
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "deepseek"}
            for m in MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _check_auth(request):
        return _err(401, "Invalid API key", "authentication_error")

    try:
        req = await request.json()
    except Exception:
        return _err(400, "Request body must be valid JSON")

    model = req.get("model") or MODELS[0]
    messages = req.get("messages") or []
    stream = bool(req.get("stream", False))

    if not messages:
        return _err(400, "`messages` is required")

    adapter = DeepSeekAdapter(DS_TOKEN, _http)

    # web_search / reasoning 由模型名或显式字段触发
    web_search = bool(req.get("web_search")) or ("search" in model.lower())
    reasoning = bool(req.get("reasoning_effort")) or ("think" in model.lower())

    try:
        headers, body, session_id, token = await adapter.prepare(
            model, messages, web_search=web_search, reasoning=reasoning
        )
    except Exception as e:
        return _err(502, f"DeepSeek prepare failed: {e}", "upstream_error")

    if not stream:
        # 非流式：聚合完整回答
        try:
            full = await _collect_stream(headers, body, session_id, token, adapter)
        except Exception as e:
            return _err(502, f"DeepSeek request failed: {e}", "upstream_error")
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
            },
        }

    # 流式：把 DeepSeek 私有 SSE 转成 OpenAI chunk
    return StreamingResponse(
        _stream_gen(headers, body, session_id, token, adapter, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _collect_stream(headers, body, session_id, token, adapter) -> str:
    """拉完整个 DeepSeek SSE，返回聚合后的正文。"""
    parts = []
    try:
        # 节流：串行化 + 最小间隔，避免触发风控
        async with _throttle:
            async with _http.stream(
                "POST",
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=headers, json=body, timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    txt = await resp.aread()
                    raise RuntimeError(f"HTTP {resp.status_code}: {txt[:300]!r}")
                # 把整个行流传给解析器，内部维护跨行状态（纯 v 追加依赖此状态）
                async for delta in iter_text_deltas_async(resp.aiter_lines()):
                    parts.append(delta)
    finally:
        await adapter._delete_session(token, session_id)

    # 空响应 = 限流特征（HTTP 200 但 content 为空）。不自动重试：
    # 限流窗口内立即重试全部无效，且增加风控权重。
    if not parts:
        raise RuntimeError(
            "DeepSeek 返回空响应，疑似触发限流。请等待约 60 秒再试，"
            "不要立即重试（窗口内重试会增加风控权重）。"
        )
    return "".join(parts)


async def _stream_gen(headers, body, session_id, token, adapter, model):
    """生成 OpenAI 风格的 SSE 流。"""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # 先发一个带 role 的空 chunk（惯例）
    yield sse_frame(json.dumps(_new_chunk(model, {"role": "assistant"}, cid=cid)))

    errorOccurred = False
    gotContent = False
    try:
        # 节流：串行化 + 最小间隔，避免触发风控
        async with _throttle:
            async with _http.stream(
                "POST",
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=headers, json=body, timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    txt = await resp.aread()
                    err = {"error": {"message": f"DeepSeek HTTP {resp.status_code}: {txt[:300]!r}",
                                     "type": "upstream_error", "param": None, "code": None}}
                    yield sse_frame(json.dumps(err))
                    errorOccurred = True
                else:
                    # 把整个行流传给解析器，内部维护跨行状态（纯 v 追加依赖此状态）
                    async for delta in iter_text_deltas_async(resp.aiter_lines()):
                        gotContent = True
                        chunk = _new_chunk(model, {"content": delta}, cid=cid)
                        yield sse_frame(json.dumps(chunk))
    except Exception as e:
        err = {"error": {"message": str(e), "type": "upstream_error", "param": None, "code": None}}
        yield sse_frame(json.dumps(err))
        errorOccurred = True
    finally:
        await adapter._delete_session(token, session_id)

    # 空响应 = 限流特征（HTTP 200 但 content 为空）。不自动重试：
    # 限流窗口内立即重试全部无效，且增加风控权重。
    if not errorOccurred and not gotContent:
        err = {"error": {"message": "DeepSeek 返回空响应，疑似触发限流。请等待约 60 秒再试，不要立即重试。",
                         "type": "rate_limited", "param": None, "code": None}}
        yield sse_frame(json.dumps(err))
        errorOccurred = True

    if not errorOccurred:
        final = _new_chunk(model, {}, finish="stop", cid=cid)
        yield sse_frame(json.dumps(final))
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    exe_dir = _os.path.dirname(_os.path.abspath(_sys.executable))
    user_cfg = _os.path.join(exe_dir, "config.yaml")

    # 首次运行（exe 旁没有用户配置）：生成默认配置并提示。
    # 内置兜底 config 位于 _MEIPASS 解压目录，可能只读，不能直接改它。
    if CONFIG_KIND == "missing":
        user_cfg = CONFIG_PATH
        with open(user_cfg, "w", encoding="utf-8") as f:
            f.write(
                "# web2api 配置文件（自动生成）\n"
                "server:\n"
                "  host: 127.0.0.1\n"
                "  port: 8080\n"
                "  api_key: \"\"\n"
                "  min_request_interval_s: 12\n"
                "deepseek:\n"
                "  token: \"PASTE_YOUR_DEEPSEEK_USER_TOKEN_HERE\"\n"
                "  models:\n"
                '    - "DeepSeek-V4.1-Flash"\n'
                '    - "deepseek-v4-pro"\n'
            )
        print(f"[首次运行] 已生成默认配置: {user_cfg}")
        print("请编辑该文件，填入你的 DeepSeek userToken（见 README），然后重新启动。")
        raise SystemExit(0)

    if "PASTE_YOUR" in DS_TOKEN:
        print("[警告] deepseek.token 仍是占位符，对话会失败。")
        print(f"请编辑 {CONFIG_PATH} 填入 token 后重启。")

    print(f"web2api listening on http://{HOST}:{PORT}")
    print(f"  config: {CONFIG_PATH}")
    print(f"  auth: {'Bearer key required' if API_KEY else 'DISABLED (local only)'}")
    print(f"  rate limit: 1 req / {MIN_INTERVAL_S}s upstream")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
