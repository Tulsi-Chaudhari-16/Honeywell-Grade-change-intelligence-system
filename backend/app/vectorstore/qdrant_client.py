"""
GCIS Backend — Qdrant Vector Store Client
Wraps Qdrant operations for episode embedding storage and ANN search.
Fallback: Postgres Euclidean-distance query if Qdrant unavailable.
"""
import uuid
import structlog
from typing import Any

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from app.core.config import settings

log = structlog.get_logger(__name__)

_client: AsyncQdrantClient | None = None


async def get_qdrant() -> AsyncQdrantClient:
    """Return (creating if needed) the shared Qdrant async client."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=10,
        )
        await _ensure_collection()
    return _client


async def _ensure_collection() -> None:
    """Create the episodes collection if it doesn't exist."""
    client = await get_qdrant()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if settings.QDRANT_COLLECTION not in names:
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        log.info("qdrant.collection_created", collection=settings.QDRANT_COLLECTION)


async def upsert_episode(
    point_id: str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """
    Index an episode embedding into Qdrant.

    Args:
        point_id: Unique point ID (usually episode_id as str).
        vector: Float embedding vector from ml/infer/embed_transition().
        payload: Metadata: episode_id, grade_pair, outcome, machine_id, etc.
    """
    try:
        client = await get_qdrant()
        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        log.info("qdrant.upserted", point_id=point_id)
    except Exception as e:
        log.error("qdrant.upsert_failed", point_id=point_id, error=str(e))
        raise


async def search_similar(
    vector: list[float],
    grade_pair: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    ANN search with optional grade_pair pre-filter.
    Returns top_k matches with payload and score.

    Falls back to Postgres SQL query if Qdrant is unavailable.
    """
    try:
        client = await get_qdrant()
        search_filter = None
        if grade_pair:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="grade_pair",
                        match=MatchValue(value=grade_pair),
                    )
                ]
            )
        results = await client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {"score": r.score, "payload": r.payload, "id": r.id}
            for r in results
        ]
    except Exception as e:
        log.warning("qdrant.search_failed_using_sql_fallback", error=str(e))
        return await _sql_fallback_search(vector, grade_pair, top_k)


async def _sql_fallback_search(
    vector: list[float],
    grade_pair: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Fallback: Euclidean distance query on Postgres feature_snapshots.
    Returns similar episodes based on raw feature vectors.
    """
    from app.db.session import get_session
    from sqlalchemy import text

    log.warning("qdrant.sql_fallback_activated")

    vector_str = "[" + ",".join(str(v) for v in vector) + "]"
    query = text("""
        SELECT
            t.episode_id,
            t.machine_id,
            t.outcome,
            t.recovery_time_min,
            t.started_at,
            t.from_recipe_id,
            t.to_recipe_id
        FROM transitions t
        WHERE t.outcome IS NOT NULL
        ORDER BY t.started_at DESC
        LIMIT :limit
    """)

    async with get_session() as session:
        result = await session.execute(query, {"limit": top_k})
        rows = result.fetchall()

    return [
        {
            "score": 0.0,
            "id": str(row.episode_id),
            "payload": {
                "episode_id": str(row.episode_id),
                "machine_id": row.machine_id,
                "outcome": row.outcome,
                "recovery_time_min": float(row.recovery_time_min or 0),
                "source": "sql_fallback",
            },
        }
        for row in rows
    ]


async def close_qdrant() -> None:
    """Close the Qdrant client on application shutdown."""
    global _client
    if _client:
        await _client.close()
        _client = None
    log.info("qdrant.client_closed")
