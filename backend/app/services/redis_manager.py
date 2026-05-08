"""Redis connection manager for real-time MARL agent state and session caching.

Falls back to an in-memory store when Redis is unavailable so local runs can
start without Docker.
"""

import redis.asyncio as redis
import json, logging
from typing import Any, Optional

logger = logging.getLogger("propos.redis")


class RedisManager:
    def __init__(self, url: str):
        self._url = url
        self._pool: Optional[redis.Redis] = None
        self._memory: dict[str, Any] = {}

    async def connect(self):
        try:
            self._pool = redis.from_url(self._url, decode_responses=True)
            await self._pool.ping()
            logger.info("Redis connected")
        except Exception as exc:
            self._pool = None
            logger.warning("Redis unavailable, using in-memory fallback: %s", exc)

    async def disconnect(self):
        if self._pool:
            await self._pool.close()

    # ── Agent state ops ───────────────────────────────────────────────
    async def set_agent_state(self, session_id: str, agent_id: str, state: dict, ttl: int = 3600):
        key = f"agent:{session_id}:{agent_id}"
        if self._pool:
            await self._pool.set(key, json.dumps(state), ex=ttl)
        else:
            self._memory[key] = json.dumps(state)

    async def get_agent_state(self, session_id: str, agent_id: str) -> Optional[dict]:
        key = f"agent:{session_id}:{agent_id}"
        raw = await self._pool.get(key) if self._pool else self._memory.get(key)
        return json.loads(raw) if raw else None

    # ── Negotiation round tracking ────────────────────────────────────
    async def push_round(self, session_id: str, round_data: dict):
        key = f"negotiation:{session_id}:rounds"
        if self._pool:
            await self._pool.rpush(key, json.dumps(round_data))
            await self._pool.expire(key, 86400)
        else:
            self._memory.setdefault(key, []).append(json.dumps(round_data))

    async def get_rounds(self, session_id: str) -> list[dict]:
        key = f"negotiation:{session_id}:rounds"
        raw = await self._pool.lrange(key, 0, -1) if self._pool else self._memory.get(key, [])
        return [json.loads(r) for r in raw]

    # ── Transparency ledger entries ───────────────────────────────────
    async def log_decision(self, trace_id: str, entry: dict):
        key = f"audit:{trace_id}"
        if self._pool:
            await self._pool.rpush(key, json.dumps(entry))
            await self._pool.expire(key, 7 * 86400)  # 7 day retention
        else:
            self._memory.setdefault(key, []).append(json.dumps(entry))

    # ── Generic cache ─────────────────────────────────────────────────
    async def cache_set(self, key: str, value: Any, ttl: int = 300):
        if self._pool:
            await self._pool.set(f"cache:{key}", json.dumps(value), ex=ttl)
        else:
            self._memory[f"cache:{key}"] = json.dumps(value)

    async def cache_get(self, key: str) -> Optional[Any]:
        raw = await self._pool.get(f"cache:{key}") if self._pool else self._memory.get(f"cache:{key}")
        return json.loads(raw) if raw else None
