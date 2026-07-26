"""
GCIS — Data Agent (Deterministic)
Kafka tag stream consumer: validates, detects staleness, backfills from Historian.
"""
import asyncio
import json
import time
from dataclasses import dataclass

import structlog
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.core.redis_client import get_lkg, set_lkg, is_alert_duplicate, publish_alert

log = structlog.get_logger(__name__)

# Expected unit map: tag_name -> expected_unit
EXPECTED_UNITS: dict[str, str] = {
    "machine_speed":       "m/min",
    "headbox_pressure":    "bar",
    "headbox_consistency": "%",
    "steam_pressure_p1":   "bar",
    "steam_pressure_p2":   "bar",
    "steam_pressure_p3":   "bar",
    "dryer_temp_zone1":    "degC",
    "dryer_temp_zone2":    "degC",
    "dryer_temp_zone3":    "degC",
    "wire_tension":        "kN/m",
    "press_nip_load":      "kN/m",
    "pulp_consistency":    "%",
    "refiner_power":       "kW",
    "broke_ratio":         "%",
}

# Value range bounds for validation (tag -> (min, max))
VALUE_BOUNDS: dict[str, tuple[float, float]] = {
    "machine_speed":       (0, 1500),
    "headbox_pressure":    (0, 10),
    "headbox_consistency": (0, 5),
    "steam_pressure_p1":   (0, 15),
    "dryer_temp_zone1":    (20, 200),
    "wire_tension":        (0, 50),
    "press_nip_load":      (0, 500),
    "pulp_consistency":    (0, 10),
    "refiner_power":       (0, 6000),
    "broke_ratio":         (0, 100),
}


@dataclass
class TagMessage:
    tag_name:   str
    value:      float
    unit:       str
    machine_id: str
    ts:         float          # Unix timestamp
    episode_id: str | None


def validate_tag(msg: TagMessage) -> list[str]:
    """Returns list of validation errors. Empty list = valid."""
    errors = []

    # Unit check
    expected = EXPECTED_UNITS.get(msg.tag_name)
    if expected and msg.unit != expected:
        errors.append(f"Unit mismatch for {msg.tag_name}: got {msg.unit}, expected {expected}")

    # Range check
    bounds = VALUE_BOUNDS.get(msg.tag_name)
    if bounds:
        lo, hi = bounds
        if not (lo <= msg.value <= hi):
            errors.append(
                f"Value out of range for {msg.tag_name}: {msg.value} not in [{lo}, {hi}]"
            )

    return errors


async def check_staleness(tag_name: str, current_ts: float, machine_id: str, fast: bool = True) -> None:
    """
    Compare last-known-good timestamp to current. Fire SensorStaleEvent if stale.
    fast=True uses STALE_TAG_FAST_SEC, else STALE_TAG_SLOW_SEC.
    """
    lkg = await get_lkg(tag_name)
    if not lkg:
        return  # first reading, no LKG yet

    threshold = settings.STALE_TAG_FAST_SEC if fast else settings.STALE_TAG_SLOW_SEC
    age = current_ts - lkg["ts"]
    if age > threshold:
        dedup_key = f"stale:{machine_id}:{tag_name}"
        if not await is_alert_duplicate(dedup_key, settings.ALERT_DEDUP_WINDOW_SEC):
            await publish_alert({
                "alert_type": "sensor_stale",
                "severity":   "warning",
                "machine_id": machine_id,
                "tag_name":   tag_name,
                "age_sec":    round(age, 1),
                "threshold":  threshold,
                "ts":         current_ts,
            })
            log.warning("data_agent.stale_tag",
                        tag=tag_name, age_sec=round(age, 1), machine=machine_id)


async def historian_backfill(tag_name: str, machine_id: str) -> float | None:
    """
    Placeholder: fetch last known value from Historian system.
    In production this calls the OPC-UA / PI System REST API.
    For hackathon demo, returns last-known-good from Redis.
    """
    lkg = await get_lkg(tag_name)
    if lkg:
        log.info("data_agent.historian_backfill", tag=tag_name, value=lkg["value"])
        return lkg["value"]
    return None


async def process_tag_message(raw: bytes) -> None:
    """Parse, validate, and store a single tag message from Kafka."""
    try:
        data = json.loads(raw.decode("utf-8"))
        msg = TagMessage(
            tag_name=data["tag_name"],
            value=float(data["value"]),
            unit=data.get("unit", ""),
            machine_id=data.get("machine_id", "unknown"),
            ts=float(data.get("ts", time.time())),
            episode_id=data.get("episode_id"),
        )

        # Validate
        errors = validate_tag(msg)
        if errors:
            log.warning("data_agent.validation_failed", tag=msg.tag_name, errors=errors)
            # Attempt historian backfill for the bad value
            backfill = await historian_backfill(msg.tag_name, msg.machine_id)
            if backfill is not None:
                msg = TagMessage(**{**msg.__dict__, "value": backfill})
            else:
                return  # Cannot proceed without a valid value

        # Check staleness
        await check_staleness(msg.tag_name, msg.ts, msg.machine_id)

        # Update last-known-good
        await set_lkg(msg.tag_name, msg.value, msg.ts)

        log.debug("data_agent.tag_processed", tag=msg.tag_name, value=msg.value)

    except Exception as e:
        log.error("data_agent.process_error", error=str(e))


async def run(state: dict) -> dict:
    """LangGraph node entry point — called by supervisor during FeatureBuilding."""
    # For graph integration, Data Agent is a background service.
    # This function is a passthrough; actual ingestion runs in start_kafka_consumer().
    return {}


async def start_kafka_consumer() -> None:
    """
    Background task: continuously consume Kafka tag topic.
    Started on FastAPI startup via lifespan event.
    """
    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC_TAGS,
        bootstrap_servers=settings.KAFKA_BROKERS,
        group_id="gcis_data_agent",
        auto_offset_reset="latest",
        value_deserializer=lambda m: m,
    )
    await consumer.start()
    log.info("data_agent.kafka_consumer_started", topic=settings.KAFKA_TOPIC_TAGS)

    try:
        async for message in consumer:
            await process_tag_message(message.value)
    finally:
        await consumer.stop()
        log.info("data_agent.kafka_consumer_stopped")
