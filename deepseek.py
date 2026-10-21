"""
DeepSeek 适配器 —— 把 OpenAI chat 请求翻译成 DeepSeek 网页版私有接口调用。

完整调用链（逐字对应原版 AFS 的 DeepSeekAdapter）：
  1. 换 token    GET  /api/v0/users/current            -> 短期 access_token（缓存1h）
  2. 建会话      POST /api/v0/chat_session/create      -> chat_session_id
  3. 取 PoW     POST /api/v0/chat/create_pow_challenge-> challenge 对象
  4. 算答案      pow.build_pow_header(challenge)        -> X-Ds-Pow-Response 头值
  5. 发对话      POST /api/v0/chat/completion           -> SSE 流
  6. 删会话      POST /api/v0/chat_session/delete       -> 清理（无状态，用完即删）
"""

import time
import json
import uuid
import random
import string

import httpx

from pow import build_pow_header

API_BASE = "https://chat.deepseek.com/api"

# 伪装成 Chrome/macOS 的浏览器请求头（逐字取自原版 FAKE_HEADERS）。
# DeepSeek 会校验部分头，缺了可能被风控。
FAKE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Origin": "https://chat.deepseek.com",
    "Referer": "https://chat.deepseek.com/",
    "Sec-Ch-Ua": '"Not/A)Brand";v="99", "Chromium";v="148"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
    "X-App-Version": "2.0.0",
    "X-Client-Locale": "zh_CN",
    "X-Client-Platform": "web",
    "x-Client-Timezone-Offset": "28800",
    "X-Client-Version": "2.0.0",
}


def _random_cookie() -> str:
    """生成一个看似正常的设备 cookie（原版 generateCookie 的简化等价物）。"""
    def rand(n, chars=string.ascii_lowercase + string.digits):
        return "".join(random.choice(chars) for _ in range(n))
    return "; ".join([
        f"_dfcaptcha={rand(10)}",
        f"Hm_lvt_{rand(8)}={int(time.time())}",
        f"sf_uuid={uuid.uuid4()}",
    ])


def messages_to_prompt(messages) -> str:
    """
    把 OpenAI 的 messages[] 压平成 DeepSeek 的单条 prompt。
    这是"无状态"设计的关键：不复用服务端会话，每次把整段历史拼成一个
    prompt 发过去，DeepSeek 侧用 <|User|>/<|Assistant|> 标记识别轮次。

    格式逐字取自原版 messagesToPrompt(isMultiTurn=false)。
    """
    # 1) 规整成 {role, text}
    proc = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = str(content or "")
        proc.append({"role": role, "text": text})
    if not proc:
        return ""

    # 2) 合并相邻同角色
    blocks = []
    cur = dict(proc[0])
    for m in proc[1:]:
        if m["role"] == cur["role"]:
            cur["text"] += "\n\n" + m["text"]
        else:
            blocks.append(cur)
            cur = dict(m)
    blocks.append(cur)

    # 3) 按角色加标记
    out = []
    for i, b in enumerate(blocks):
        if b["role"] == "assistant":
            out.append(f"<|Assistant|>{b['text']}<|end of sentence|>")
        elif b["role"] in ("user", "system"):
            # 首块若是 user/system 直接裸文本，其余加 <|User|> 前缀
            out.append(b["text"] if i == 0 else f"<|User|>{b['text']}")
        else:
            out.append(b["text"])
    return "".join(out)


def resolve_model_options(model: str, web_search: bool = False, reasoning: bool = False):
    """模型名 -> (model_type, search_enabled, thinking_enabled)。逐字取自原版。"""
    low = (model or "").lower()
    is_pro = "deepseek-v4-pro" in low or "expert" in low
    is_search = "search" in low
    is_think = "think" in low or "r1" in low or "reasoner" in low
    return {
        "model_type": "expert" if is_pro else "default",
        "search_enabled": bool(web_search) or is_search,
        "thinking_enabled": bool(reasoning) or is_think,
    }


class DeepSeekAdapter:
    def __init__(self, token: str, client: httpx.AsyncClient):
        self.user_token = token          # 用户填的长期 userToken
        self.client = client
        self._access_token = None
        self._access_token_exp = 0.0

    # ---- 1. 换短期 access_token（缓存 ~1h）----
    async def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._access_token_exp:
            return self._access_token
        r = await self.client.get(
            f"{API_BASE}/v0/users/current",
            headers={"Authorization": f"Bearer {self.user_token}", **FAKE_HEADERS},
            timeout=15,
        )
        if r.status_code in (401, 403):
            raise RuntimeError("DeepSeek token 无效或已过期，请重新获取 userToken")
        if r.status_code != 200:
            raise RuntimeError(f"换取 access_token 失败: HTTP {r.status_code}")
        data = r.json()
        biz = (data.get("data") or {}).get("biz_data") or data.get("biz_data") or {}
        at = biz.get("token")
        if not at:
            raise RuntimeError(f"换取 access_token 失败: {data.get('msg') or data}")
        self._access_token = at
        self._access_token_exp = time.time() + 3500  # ~58min
        return at

    # ---- 2. 建会话 ----
    async def _create_session(self, token: str) -> str:
        r = await self.client.post(
            f"{API_BASE}/v0/chat_session/create",
            json={},
            headers={"Authorization": f"Bearer {token}", **FAKE_HEADERS,
                     "Cookie": _random_cookie()},
            timeout=15,
        )
        data = r.json()
        biz = (data.get("data") or {}).get("biz_data") or data.get("biz_data") or {}
        sid = (((biz.get("chat_session") or {}).get("id")))
        if r.status_code != 200 or not sid:
            raise RuntimeError(f"创建会话失败: {data.get('msg') or r.status_code}")
        return sid

    # ---- 3. 取 PoW challenge ----
    async def _get_pow_challenge(self, token: str) -> dict:
        r = await self.client.post(
            f"{API_BASE}/v0/chat/create_pow_challenge",
            json={"target_path": "/api/v0/chat/completion"},
            headers={"Authorization": f"Bearer {token}", **FAKE_HEADERS},
            timeout=15,
        )
        data = r.json()
        biz = (data.get("data") or {}).get("biz_data") or data.get("biz_data") or {}
        ch = biz.get("challenge")
        if r.status_code != 200 or not ch:
            raise RuntimeError(f"获取 PoW challenge 失败: {data.get('msg') or r.status_code}")
        return ch

    # ---- 6. 删会话（尽力而为）----
    async def _delete_session(self, token: str, session_id: str):
        try:
            await self.client.post(
                f"{API_BASE}/v0/chat_session/delete",
                json={"chat_session_id": session_id},
                headers={"Authorization": f"Bearer {token}", **FAKE_HEADERS},
                timeout=15,
            )
        except Exception:
            pass

    async def prepare(self, model: str, messages, web_search=False, reasoning=False):
        """完成 换token->建会话->取PoW->算答案，返回 (headers, body, session_id)。"""
        token = await self._get_access_token()
        session_id = await self._create_session(token)
        challenge = await self._get_pow_challenge(token)
        pow_header = build_pow_header(challenge, "/api/v0/chat/completion")

        prompt = messages_to_prompt(messages)
        opts = resolve_model_options(model, web_search, reasoning)

        body = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "model_type": opts["model_type"],
            "ref_file_ids": [],
            "search_enabled": opts["search_enabled"],
            "thinking_enabled": opts["thinking_enabled"],
            "preempt": False,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            **FAKE_HEADERS,
            "Referer": f"https://chat.deepseek.com/a/chat/s/{session_id}",
            "Cookie": _random_cookie(),
            "X-Ds-Pow-Response": pow_header,
        }
        return headers, body, session_id, token
