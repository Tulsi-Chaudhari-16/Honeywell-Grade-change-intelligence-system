"""
GCIS Backend — Async Redis Client
Shared connection pool for:
- Last-known-good tag values (Data Agent)
- LangGraph checkpoint backend (Supervisor)
- Pub/sub for inter-agent signaling
- Alert deduplication keys
"""
import structlog
import redis.asyncio as aioredis
from redis.asyncio import Redis, ConnectionPool

from app.core.config import settings

log = structlog.get_logger(__name__)

_pool: ConnectionPool | None = None
_client: Redis | None = None


async def get_redis_pool() -> ConnectionPool:
    """Return (creating if needed) the shared async Redis connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=50,
            decode_responses=True,
        )
        log.info("redis.pool_created", url=settings.REDIS_URL)
    return _pool


async def get_redis() -> Redis:
    """Return a Redis client backed by the shared pool."""
    global _client
    if _client is None:
        pool = await get_redis_pool()
        _client = aioredis.Redis(connection_pool=pool)
    return _client


async def close_redis() -> None:
    """Close the pool on application shutdown."""
    global _pool, _client
    if _client:
        await _client.aclose()
        _client = None
    if _pool:
        await _pool.aclose()
        _pool = None
    log.info("redis.pool_closed")


# ─── Helper: last-known-good tag values ──────────────────────────────────────

async def set_lkg(tag_name: str, value: float, ts: float) -> None:
    """Store last-known-good value for a sensor tag."""
    r = await get_redis()
    await r.hset(f"lkg:{tag_name}", mapping={"value": value, "ts": ts})


async def get_lkg(tag_name: str) -> dict | None:
    """Retrieve last-known-good value and timestamp for a sensor tag."""
    r = await get_redis()
    data = await r.hgetall(f"lkg:{tag_name}")
    if not data:
        return None
    return {"value": float(data["value"]), "ts": float(data["ts"])}


# ─── Helper: alert deduplication ─────────────────────────────────────────────

async def is_alert_duplicate(dedup_key: str, window_sec: int) -> bool:
    """
    Check if alert was recently fired. Sets key with TTL if not.
    Returns True if duplicate (should suppress), False if new.
    """
    r = await get_redis()
    key = f"alert_dedup:{dedup_key}"
    result = await r.set(key, 1, ex=window_sec, nx=True)
    return result is None  # nx=True means set only if not exists; None = already existed


# ─── Helper: episode state pub/sub ───────────────────────────────────────────

async def publish_episode_state(episode_id: str, state: dict) -> None:
    """Publish updated episode state to Redis channel for WebSocket pickup."""
    import json
    r = await get_redis()
    await r.publish(f"episode:{episode_id}", json.dumps(state))


async def publish_alert(alert: dict) -> None:
    """Publish alert to Redis global alert channel."""
    import json
    r = await get_redis()
    await r.publish("alerts:global", json.dumps(alert))
