"""
GCIS Backend — Main Application Entrypoint
Registers all routers, configures CORS, and handles startup/shutdown events.
"""
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.redis_client import close_redis
from app.vectorstore.qdrant_client import close_qdrant
from app.agents.data_agent import start_kafka_consumer

# Routers
from app.api.episodes import router as episodes_router
from app.api.recommendations import router as recommendations_router
from app.api.recipes import router as recipes_router
from app.api.alerts import router as alerts_router
from app.api.simulate import router as simulate_router
from app.api.websocket import router as ws_router
from app.api.chat import router as chat_router

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI."""
    log.info("gcis_backend.starting")
    
    # Start Kafka consumer in the background
    import asyncio
    consumer_task = asyncio.create_task(start_kafka_consumer())
    
    yield
    
    log.info("gcis_backend.shutting_down")
    # Graceful shutdown
    consumer_task.cancel()
    await close_redis()
    await close_qdrant()


app = FastAPI(
    title="GCIS API",
    description="Agent Orchestration & Real-time Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes. In prod, lock this down.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST routers
app.include_router(episodes_router, prefix="/api/v1/episodes", tags=["Episodes"])
app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(recipes_router, prefix="/api/v1/recipes", tags=["Recipes"])
app.include_router(alerts_router, prefix="/api/v1/alerts", tags=["Alerts"])
app.include_router(simulate_router, prefix="/api/v1/simulate", tags=["Simulation"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])

# Register WebSocket router
app.include_router(ws_router, tags=["WebSockets"])


@app.get("/api/v1/system/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "provider": settings.LLM_PROVIDER}

@app.get("/api/v1/system/drift", tags=["System"])
async def drift_metrics():
    """Placeholder for drift metrics."""
    return {"kl_divergence": 0.05, "data_drift_detected": False}
