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


# ============================================================================
# Anthropic Messages API 兼容层 (支持 Claude Code / 标准 Anthropic 客户端)
# ============================================================================

def _anthropic_err(status: int, message: str, err_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _anthropic_sse_frame(event_type: str, data_dict: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data_dict, ensure_ascii=False)}\n\n"


class ToolCallStreamParser:
    """流式工具调用解析器，识别模型回复中的 <tool_call>{...}</tool_call> 标记。"""
    START_TAG = "<tool_call>"
    END_TAG = "</tool_call>"

    def __init__(self):
        self.buffer = ""
        self.in_tool = False

    def feed(self, delta: str):
        self.buffer += delta
        events = []

        while self.buffer:
            if not self.in_tool:
                idx = self.buffer.find(self.START_TAG)
                if idx != -1:
                    if idx > 0:
                        events.append(("text", self.buffer[:idx]))
                    self.buffer = self.buffer[idx + len(self.START_TAG):]
                    self.in_tool = True
                    continue
                else:
                    matched_prefix_len = 0
                    for k in range(min(len(self.buffer), len(self.START_TAG) - 1), 0, -1):
                        if self.START_TAG.startswith(self.buffer[-k:]):
                            matched_prefix_len = k
                            break
                    if matched_prefix_len > 0:
                        safe_text = self.buffer[:-matched_prefix_len]
                        if safe_text:
                            events.append(("text", safe_text))
                        self.buffer = self.buffer[-matched_prefix_len:]
                        break
                    else:
                        events.append(("text", self.buffer))
                        self.buffer = ""
                        break
            else:
                idx = self.buffer.find(self.END_TAG)
                if idx != -1:
                    raw_tool = self.buffer[:idx].strip()
                    self.buffer = self.buffer[idx + len(self.END_TAG):]
                    self.in_tool = False
                    try:
                        obj = json.loads(raw_tool)
                        name = obj.get("name", "unknown")
                        args = obj.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        tid = f"toolu_{uuid.uuid4().hex[:20]}"
                        events.append(("tool_use", tid, name, args))
                    except Exception:
                        events.append(("text", f"{self.START_TAG}{raw_tool}{self.END_TAG}"))
                    continue
                else:
                    break

        return events

    def flush(self):
        events = []
        if self.in_tool:
            events.append(("text", f"{self.START_TAG}{self.buffer}"))
        elif self.buffer:
            events.append(("text", self.buffer))
        self.buffer = ""
        self.in_tool = False
        return events


def _anthropic_to_adapted_messages(req: dict) -> list:
    """把 Anthropic 请求的 system/tools/messages 翻译成适用于 messages_to_prompt 的消息列表。"""
    system_parts = []
    raw_system = req.get("system")
    if isinstance(raw_system, str) and raw_system.strip():
        system_parts.append(raw_system.strip())
    elif isinstance(raw_system, list):
        for block in raw_system:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    system_parts.append(t)

    # 注入工具说明与 Schema
    tools = req.get("tools") or []
    if tools:
        tool_desc = json.dumps(tools, ensure_ascii=False, indent=2)
        tool_prompt = (
            "# 工具调用协议\n"
            "你可以调用以下工具。需要调用时，请在回复中输出如下格式的标记（支持在一轮回复中连续输出多个）：\n"
            "<tool_call>\n"
            '{"name": "工具名称", "arguments": {"参数名": "参数值"}}\n'
            "</tool_call>\n\n"
            "规则：\n"
            "- 必须严格使用 <tool_call> 和 </tool_call> 标签包裹。\n"
            '- 标签内部必须是合法的 JSON 对象，且包含 "name" 和 "arguments" 字段。\n'
            "- 可以在 <tool_call> 之前或之后提供简要说明，收到工具结果后继续回答。\n\n"
            f"可用工具定义（JSON Schema）：\n{tool_desc}"
        )
        system_parts.append(tool_prompt)

        tool_choice = req.get("tool_choice")
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "any":
                system_parts.append("[系统要求]：本轮回答必须调用工具。")
            elif tc_type == "tool":
                tname = tool_choice.get("name", "")
                system_parts.append(f"[系统要求]：本轮回答必须调用工具 {tname}。")

    adapted_messages = []
    if system_parts:
        adapted_messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for msg in req.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text_parts = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    call_dict = {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                    text_parts.append(
                        f"\n<tool_call>\n{json.dumps(call_dict, ensure_ascii=False)}\n</tool_call>\n"
                    )
                elif btype == "tool_result":
                    tid = block.get("tool_use_id", "")
                    is_err = block.get("is_error", False)
                    err_hint = " [执行出错]" if is_err else ""
                    res_c = block.get("content", "")
                    if isinstance(res_c, list):
                        res_texts = [
                            sub.get("text", "") for sub in res_c if isinstance(sub, dict) and sub.get("type") == "text"
                        ]
                        res_c = "\n".join(res_texts)
                    text_parts.append(f"\n[工具返回{err_hint} id={tid}]:\n{res_c}\n")
        adapted_messages.append({"role": role, "content": "".join(text_parts).strip()})

    return adapted_messages


async def _anthropic_collect_stream(headers, body, session_id, token, adapter):
    """收集完整响应并解析工具调用，返回 (content_blocks, stop_reason, total_chars)。"""
    parts = []
    try:
        async with _throttle:
            async with _http.stream(
                "POST",
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=headers, json=body, timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    txt = await resp.aread()
                    raise RuntimeError(f"HTTP {resp.status_code}: {txt[:300]!r}")
                async for delta in iter_text_deltas_async(resp.aiter_lines()):
                    parts.append(delta)
    finally:
        await adapter._delete_session(token, session_id)

    if not parts:
        raise RuntimeError(
            "DeepSeek 返回空响应，疑似触发限流。请等待约 60 秒再试，"
            "不要立即重试（窗口内重试会增加风控权重）。"
        )

    parser = ToolCallStreamParser()
    events = []
    for p in parts:
        events.extend(parser.feed(p))
    events.extend(parser.flush())

    content_blocks = []
    has_tool_use = False
    total_chars = 0
    accumulated_text = []

    for ev in events:
        if ev[0] == "text":
            txt = ev[1]
            if txt:
                total_chars += len(txt)
                accumulated_text.append(txt)
        elif ev[0] == "tool_use":
            has_tool_use = True
            if accumulated_text:
                content_blocks.append({"type": "text", "text": "".join(accumulated_text)})
                accumulated_text = []
            _, tid, name, args = ev
            content_blocks.append({"type": "tool_use", "id": tid, "name": name, "input": args})

    if accumulated_text:
        content_blocks.append({"type": "text", "text": "".join(accumulated_text)})

    stop_reason = "tool_use" if has_tool_use else "end_turn"
    return content_blocks, stop_reason, total_chars


async def _anthropic_stream_gen(headers, body, session_id, token, adapter, model: str):
    """把 DeepSeek 私有 SSE 转换为 Anthropic 标准 SSE 事件流。"""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _anthropic_sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    })

    parser = ToolCallStreamParser()
    block_index = 0
    text_block_open = False
    has_tool_use = False
    total_chars = 0
    error_occurred = False
    got_content = False

    try:
        async with _throttle:
            async with _http.stream(
                "POST",
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=headers, json=body, timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    txt = await resp.aread()
                    err = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"DeepSeek HTTP {resp.status_code}: {txt[:300]!r}",
                        },
                    }
                    yield _anthropic_sse_frame("error", err)
                    error_occurred = True
                else:
                    async for delta in iter_text_deltas_async(resp.aiter_lines()):
                        got_content = True
                        for ev in parser.feed(delta):
                            etype = ev[0]
                            if etype == "text":
                                txt = ev[1]
                                if not txt:
                                    continue
                                total_chars += len(txt)
                                if not text_block_open:
                                    yield _anthropic_sse_frame("content_block_start", {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                    text_block_open = True
                                yield _anthropic_sse_frame("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": txt},
                                })
                            elif etype == "tool_use":
                                _, tid, name, args = ev
                                has_tool_use = True
                                if text_block_open:
                                    yield _anthropic_sse_frame("content_block_stop", {
                                        "type": "content_block_stop",
                                        "index": block_index,
                                    })
                                    text_block_open = False
                                    block_index += 1

                                yield _anthropic_sse_frame("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}},
                                })
                                yield _anthropic_sse_frame("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(args, ensure_ascii=False),
                                    },
                                })
                                yield _anthropic_sse_frame("content_block_stop", {
                                    "type": "content_block_stop",
                                    "index": block_index,
                                })
                                block_index += 1

                    # 刷出缓冲区残留
                    for ev in parser.flush():
                        etype = ev[0]
                        if etype == "text":
                            txt = ev[1]
                            if txt:
                                total_chars += len(txt)
                                if not text_block_open:
                                    yield _anthropic_sse_frame("content_block_start", {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                    text_block_open = True
                                yield _anthropic_sse_frame("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": txt},
                                })
                        elif etype == "tool_use":
                            _, tid, name, args = ev
                            has_tool_use = True
                            if text_block_open:
                                yield _anthropic_sse_frame("content_block_stop", {
                                    "type": "content_block_stop",
                                    "index": block_index,
                                })
                                text_block_open = False
                                block_index += 1

                            yield _anthropic_sse_frame("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}},
                            })
                            yield _anthropic_sse_frame("content_block_delta", {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(args, ensure_ascii=False),
                                },
                            })
                            yield _anthropic_sse_frame("content_block_stop", {
                                "type": "content_block_stop",
                                "index": block_index,
                            })
                            block_index += 1

    except Exception as e:
        err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
        yield _anthropic_sse_frame("error", err)
        error_occurred = True
    finally:
        await adapter._delete_session(token, session_id)

    if not error_occurred and not got_content:
        err = {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "DeepSeek 返回空响应，疑似触发限流。请等待约 60 秒再试，不要立即重试。",
            },
        }
        yield _anthropic_sse_frame("error", err)
        error_occurred = True

    if not error_occurred:
        if text_block_open:
            yield _anthropic_sse_frame("content_block_stop", {
                "type": "content_block_stop",
                "index": block_index,
            })
            block_index += 1

        stop_reason = "tool_use" if has_tool_use else "end_turn"
        est_tokens = max(1, total_chars // 4)
        yield _anthropic_sse_frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": est_tokens},
        })
        yield _anthropic_sse_frame("message_stop", {"type": "message_stop"})


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    if not _check_auth(request):
        return _anthropic_err(401, "Invalid API key", "authentication_error")

    try:
        req = await request.json()
    except Exception:
        return _anthropic_err(400, "Request body must be valid JSON")

    messages = req.get("messages") or []
    if not messages:
        return _anthropic_err(400, "`messages` is required")

    raw_model = req.get("model") or "claude-opus-5-5"
    # 若客户端指定的模型在已知配置列表中，优先使用；否则回退到默认的 MODELS[0]
    upstream_model = raw_model if raw_model in MODELS else MODELS[0]
    stream = bool(req.get("stream", False))

    adapted_messages = _anthropic_to_adapted_messages(req)
    adapter = DeepSeekAdapter(DS_TOKEN, _http)

    web_search = bool(req.get("web_search")) or ("search" in raw_model.lower())
    reasoning = bool(req.get("thinking")) or ("think" in raw_model.lower())

    try:
        headers, body, session_id, token = await adapter.prepare(
            upstream_model, adapted_messages, web_search=web_search, reasoning=reasoning
        )
    except Exception as e:
        return _anthropic_err(502, f"DeepSeek prepare failed: {e}", "api_error")

    if not stream:
        try:
            content_blocks, stop_reason, total_chars = await _anthropic_collect_stream(
                headers, body, session_id, token, adapter
            )
        except Exception as e:
            return _anthropic_err(502, f"DeepSeek request failed: {e}", "api_error")

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": raw_model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": max(1, total_chars // 4),
            },
        }

    return StreamingResponse(
        _anthropic_stream_gen(headers, body, session_id, token, adapter, raw_model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    if not _check_auth(request):
        return _anthropic_err(401, "Invalid API key", "authentication_error")

    try:
        req = await request.json()
    except Exception:
        return _anthropic_err(400, "Request body must be valid JSON")

    total_chars = 0
    sys_prompt = req.get("system")
    if isinstance(sys_prompt, str):
        total_chars += len(sys_prompt)
    elif isinstance(sys_prompt, list):
        for b in sys_prompt:
            if isinstance(b, dict):
                total_chars += len(str(b.get("text", "")))
    for m in req.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    total_chars += len(str(b.get("text", ""))) + len(str(b.get("input", "")))
    for t in req.get("tools", []):
        total_chars += len(json.dumps(t, ensure_ascii=False))

    est_tokens = max(1, total_chars // 4)
    return {"input_tokens": est_tokens}


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
