"""Small retry-with-backoff helper for Gemini rate-limit (429) errors,
which are common on free-tier API keys."""
from __future__ import annotations

import asyncio
import re

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _suggested_delay(exc: Exception, fallback: float) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1
    return fallback


async def with_retry_async(coro_factory, *, attempts: int = 3, base_delay: float = 5.0):
    """Call an async factory, retrying on Gemini rate-limit errors with backoff."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limit(exc) or attempt == attempts - 1:
                raise
            delay = _suggested_delay(exc, base_delay * (attempt + 1))
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover


def with_retry_sync(fn, *, attempts: int = 3, base_delay: float = 5.0):
    """Call a sync function, retrying on Gemini rate-limit errors with backoff."""
    import time

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limit(exc) or attempt == attempts - 1:
                raise
            delay = _suggested_delay(exc, base_delay * (attempt + 1))
            time.sleep(delay)
    raise last_exc  # pragma: no cover
