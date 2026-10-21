"""
上游节流 —— 串行化 + 最小间隔，避免并发/高频触发 DeepSeek 网页端风控。

实测依据（来自 Trim-main 踩坑记录，非猜测）：
  - 单分钟 >10 次请求 -> 账号级禁言（连网页手动访问都受限）
  - 并发请求 -> 触发风控
  - 限流窗口内立即重试 -> 全部无效且增加风控权重

因此这里做两件事：
  1. 串行：同一时刻只允许一个上游请求在飞（asyncio.Lock），杜绝并发。
  2. 最小间隔：两次上游请求的“发起时刻”间隔不小于 min_interval，
     默认 12s ≈ 5 次/分钟，远低于 10 次/分钟的禁言线。

代价：多步/高频任务会变慢。这是避免禁言的必要取舍，不是缺陷。
"""

import asyncio
import time


class UpstreamThrottle:
    """用法：
        async with throttle:
            ... 发起上游请求 ...
    进入时排队 + 等待间隔；整个上游交互期间持有锁（串行）。
    """

    def __init__(self, min_interval_s: float = 12.0):
        self.min_interval = max(0.0, float(min_interval_s))
        self._lock = asyncio.Lock()
        self._last_start = 0.0

    async def __aenter__(self):
        await self._lock.acquire()
        try:
            now = time.monotonic()
            wait = self._last_start + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_start = time.monotonic()
            return self
        except BaseException:
            self._lock.release()
            raise

    async def __aexit__(self, *exc):
        self._lock.release()
        return False
