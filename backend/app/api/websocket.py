"""
GCIS Backend — WebSocket API
Manages live connections for episode state and global alerts.
Uses Redis pub/sub internally to broadcast messages.
"""
import asyncio
import json
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis

from app.core.config import settings

log = structlog.get_logger(__name__)

router = APIRouter()


async def redis_listener(websocket: WebSocket, channels: list[str]):
    """Listen to Redis channels and forward messages to the WebSocket."""
    redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(*channels)
    
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error("websocket.redis_listener_error", error=str(e))
    finally:
        await pubsub.unsubscribe(*channels)
        await pubsub.close()
        await redis.aclose()


@router.websocket("/ws/episodes/{episode_id}")
async def websocket_episode_endpoint(websocket: WebSocket, episode_id: str):
    """Live dashboard state stream for a specific episode."""
    await websocket.accept()
    log.info("websocket.client_connected", episode_id=episode_id)
    
    channel = f"episode:{episode_id}"
    listener_task = asyncio.create_task(redis_listener(websocket, [channel]))
    
    try:
        while True:
            # Keep connection alive and handle client disconnects
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        log.info("websocket.client_disconnected", episode_id=episode_id)
    except Exception as e:
        log.error("websocket.error", error=str(e), episode_id=episode_id)
    finally:
        listener_task.cancel()


@router.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """Global alert stream."""
    await websocket.accept()
    log.info("websocket.alert_client_connected")
    
    channel = "alerts:global"
    listener_task = asyncio.create_task(redis_listener(websocket, [channel]))
    
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        log.info("websocket.alert_client_disconnected")
    except Exception as e:
        log.error("websocket.alert_error", error=str(e))
    finally:
        listener_task.cancel()
