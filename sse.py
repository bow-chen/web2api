"""
DeepSeek SSE 流解析。

DeepSeek /v0/chat/completion 返回私有格式的 SSE，不是 OpenAI 格式。
本模块把它归一成"正文增量生成器"，逐次 yield 正文片段。

真实返回格式（抓包确认）——混合三种结构：

1) 初始块: {"v":{"response":{...,"fragments":[{"type":"ANSWER","content":"今天"},...]}}}
2) 增量块(带路径): {"p":"response/fragments/-1/content","o":"APPEND","v":"天气"}
3) 连续追加(纯v): {"v":"很好"}          <- 没有 p，延续上一条路径
4) 收尾块: {"p":"response/status","o":"SET","v":"FINISHED"}

要点：
  - 只有 v 字段的行（无 p）是"上一条路径的继续追加"，必须有状态才能正确归位。
  - 路径以 /content 结尾 -> 正式回答；以 /thinking 结尾 -> 思考过程（默认丢弃）。
  - 兼容旧的整块 fragments 列表结构（原版两种结构 A/B）。

本模块 yield 的每个元素是一段正文增量（str）。
"""

import json
import re
from typing import Iterator

# 路径形如 response/fragments/-1/content 或 response/fragments/0/thinking
_PATH_RE = re.compile(r"^response/fragments/.*/(content|thinking)$")


def iter_text_deltas(raw_lines_iter, emit_thinking: bool = False) -> Iterator[str]:
    """
    入参 raw_lines_iter：产出 bytes/str 行的迭代器（已按 \\n 切好）。
    产出正文增量字符串。
    """
    current_path = ""          # 记住上一条增量路径，供"纯 v 追加"归位
    for line in raw_lines_iter:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip() or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except Exception:
            continue

        p = chunk.get("p")
        v = chunk.get("v")
        op = chunk.get("o")

        # --- 1) 带路径的增量块: {"p":"response/fragments/-1/content","o":"APPEND","v":"天气"}
        if isinstance(p, str):
            m = _PATH_RE.match(p)
            if m and isinstance(v, str) and op == "APPEND":
                current_path = m.group(1)
                if current_path == "content" or emit_thinking:
                    if v:
                        yield v
                continue
            # 路径是 fragments 列表整块（旧结构 B 兼容）
            if p == "response/fragments" and isinstance(v, list):
                for frag in v:
                    t = (frag or {}).get("type")
                    c = (frag or {}).get("content")
                    if c and (t in ("ANSWER", "RESPONSE") or (emit_thinking and t == "THINK")):
                        yield c
                continue
            # 其余 path 事件（status/batch/search 等）不产出正文
            continue

        # --- 2) 纯 v 追加（无 p）: {"v":"很好"} —— 延续当前路径
        if isinstance(v, str):
            if current_path == "content" or emit_thinking:
                if v:
                    yield v
            continue

        # --- 3) 旧结构 A: {"v":{"response":{"fragments":[...]}}}
        if isinstance(v, dict):
            resp = v.get("response")
            if isinstance(resp, dict):
                frags = resp.get("fragments")
                if isinstance(frags, list):
                    for f in frags:
                        t = (f or {}).get("type")
                        c = (f or {}).get("content")
                        if c and (t in ("ANSWER", "RESPONSE") or (emit_thinking and t == "THINK")):
                            yield c
            continue


def sse_frame(text: str) -> str:
    """把一段文本包成 OpenAI 风格的一个 SSE 帧。"""
    return f"data: {text}\n\n"


async def iter_text_deltas_async(raw_lines_iter, emit_thinking: bool = False):
    """
    async 版本：raw_lines_iter 是一个 async 迭代器（如 httpx resp.aiter_lines()）。
    与 iter_text_deltas 逻辑一致，但内部 async for，可边读边 yield（流式友好）。
    """
    current_path = ""
    async for line in raw_lines_iter:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip() or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except Exception:
            continue

        p = chunk.get("p")
        v = chunk.get("v")
        op = chunk.get("o")

        if isinstance(p, str):
            m = _PATH_RE.match(p)
            if m and isinstance(v, str) and op == "APPEND":
                current_path = m.group(1)
                if current_path == "content" or emit_thinking:
                    if v:
                        yield v
                continue
            if p == "response/fragments" and isinstance(v, list):
                for frag in v:
                    t = (frag or {}).get("type")
                    c = (frag or {}).get("content")
                    if c and (t in ("ANSWER", "RESPONSE") or (emit_thinking and t == "THINK")):
                        yield c
                continue
            continue

        if isinstance(v, str):
            if current_path == "content" or emit_thinking:
                if v:
                    yield v
            continue

        if isinstance(v, dict):
            resp = v.get("response")
            if isinstance(resp, dict):
                frags = resp.get("fragments")
                if isinstance(frags, list):
                    for f in frags:
                        t = (f or {}).get("type")
                        c = (f or {}).get("content")
                        if c and (t in ("ANSWER", "RESPONSE") or (emit_thinking and t == "THINK")):
                            yield c
            continue

