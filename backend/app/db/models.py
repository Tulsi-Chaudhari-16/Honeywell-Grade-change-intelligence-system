"""
GCIS Backend — SQLAlchemy ORM Models
Matches 001_initial_schema.sql exactly.
"""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger, Boolean, Column, ForeignKey, Integer,
    Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─── recipes ──────────────────────────────────────────────────────────────────
class Recipe(Base):
    __tablename__ = "recipes"

    recipe_id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    grade_code          = Column(String(32), nullable=False)
    grade_name          = Column(String(128))
    target_basis_weight = Column(Numeric(8, 2))
    target_moisture     = Column(Numeric(6, 3))
    target_ash          = Column(Numeric(6, 3))
    target_caliper      = Column(Numeric(8, 3))
    created_at          = Column(TIMESTAMPTZ, default=datetime.utcnow)

    from_transitions    = relationship("Transition", foreign_keys="Transition.from_recipe_id", back_populates="from_recipe")
    to_transitions      = relationship("Transition", foreign_keys="Transition.to_recipe_id", back_populates="to_recipe")


# ─── machine_limits ───────────────────────────────────────────────────────────
class MachineLimit(Base):
    __tablename__ = "machine_limits"
    __table_args__ = (UniqueConstraint("machine_id", "variable_name"),)

    limit_id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    machine_id         = Column(String(32), nullable=False)
    variable_name      = Column(String(64), nullable=False)
    min_value          = Column(Numeric)
    max_value          = Column(Numeric)
    max_rate_of_change = Column(Numeric)
    unit               = Column(String(16))


# ─── operators ────────────────────────────────────────────────────────────────
class Operator(Base):
    __tablename__ = "operators"

    operator_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    display_name = Column(String(128))
    shift        = Column(String(16))
    created_at   = Column(TIMESTAMPTZ, default=datetime.utcnow)

    feedback     = relationship("Feedback", back_populates="operator")
    alerts_acked = relationship("Alert", back_populates="acknowledged_by_op")


# ─── transitions ──────────────────────────────────────────────────────────────
class Transition(Base):
    __tablename__ = "transitions"

    episode_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    machine_id          = Column(String(32), nullable=False)
    from_recipe_id      = Column(UUID(as_uuid=True), ForeignKey("recipes.recipe_id"))
    to_recipe_id        = Column(UUID(as_uuid=True), ForeignKey("recipes.recipe_id"))
    started_at          = Column(TIMESTAMPTZ, nullable=False)
    ended_at            = Column(TIMESTAMPTZ)
    outcome             = Column(String(16))          # success | offspec | unconfirmed
    recovery_time_min   = Column(Numeric)
    final_basis_weight  = Column(Numeric(8, 2))
    final_moisture      = Column(Numeric(6, 3))
    final_ash           = Column(Numeric(6, 3))
    embedding_id        = Column(String(64))
    current_state       = Column(String(32), default="Monitoring")
    created_at          = Column(TIMESTAMPTZ, default=datetime.utcnow)

    from_recipe    = relationship("Recipe", foreign_keys=[from_recipe_id], back_populates="from_transitions")
    to_recipe      = relationship("Recipe", foreign_keys=[to_recipe_id], back_populates="to_transitions")
    predictions    = relationship("Prediction", back_populates="transition")
    recommendations = relationship("Recommendation", back_populates="transition")
    alerts         = relationship("Alert", back_populates="transition")
    feature_snapshots = relationship("FeatureSnapshot", back_populates="transition")


# ─── predictions ──────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id        = Column(UUID(as_uuid=True), ForeignKey("transitions.episode_id"))
    p_offspec         = Column(Numeric(5, 4), nullable=False)
    trajectory        = Column(JSONB)
    eta_stabilize_min = Column(Numeric(6, 2))
    confidence        = Column(Numeric(5, 4))
    model_version     = Column(String(64))
    degraded_mode     = Column(Boolean, default=False)
    created_at        = Column(TIMESTAMPTZ, default=datetime.utcnow)

    transition      = relationship("Transition", back_populates="predictions")
    recommendations = relationship("Recommendation", back_populates="prediction")


# ─── feature_snapshots ────────────────────────────────────────────────────────
class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    snapshot_id    = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id     = Column(UUID(as_uuid=True), ForeignKey("transitions.episode_id"))
    feature_vector = Column(JSONB, nullable=False)
    window_start   = Column(TIMESTAMPTZ)
    window_end     = Column(TIMESTAMPTZ)
    created_at     = Column(TIMESTAMPTZ, default=datetime.utcnow)

    transition = relationship("Transition", back_populates="feature_snapshots")


# ─── recommendations ──────────────────────────────────────────────────────────
class Recommendation(Base):
    __tablename__ = "recommendations"

    recommendation_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id         = Column(UUID(as_uuid=True), ForeignKey("transitions.episode_id"))
    prediction_id      = Column(UUID(as_uuid=True), ForeignKey("predictions.prediction_id"))
    variable_name      = Column(String(64))
    direction          = Column(String(16))
    proposed_value     = Column(Numeric)
    clamped_value      = Column(Numeric)
    confidence         = Column(Numeric(5, 4))
    historical_support = Column(JSONB)
    safety_status      = Column(String(16))    # approved | clamped | rejected
    rejection_reason   = Column(Text)
    created_at         = Column(TIMESTAMPTZ, default=datetime.utcnow)

    transition = relationship("Transition", back_populates="recommendations")
    prediction = relationship("Prediction", back_populates="recommendations")
    feedback   = relationship("Feedback", back_populates="recommendation")


# ─── feedback ─────────────────────────────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id = Column(UUID(as_uuid=True), ForeignKey("recommendations.recommendation_id"))
    operator_id       = Column(UUID(as_uuid=True), ForeignKey("operators.operator_id"))
    action            = Column(String(16), nullable=False)  # accepted | rejected | modified
    modified_value    = Column(Numeric)
    note              = Column(Text)
    ts                = Column(TIMESTAMPTZ, default=datetime.utcnow)

    recommendation = relationship("Recommendation", back_populates="feedback")
    operator       = relationship("Operator", back_populates="feedback")


# ─── alerts ───────────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    alert_id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id      = Column(UUID(as_uuid=True), ForeignKey("transitions.episode_id"))
    severity        = Column(String(16), nullable=False)  # critical | warning | info
    alert_type      = Column(String(64), nullable=False)
    message         = Column(Text)
    ts              = Column(TIMESTAMPTZ, default=datetime.utcnow)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("operators.operator_id"))
    acknowledged_at = Column(TIMESTAMPTZ)
    dedup_key       = Column(String(128))

    transition           = relationship("Transition", back_populates="alerts")
    acknowledged_by_op   = relationship("Operator", back_populates="alerts_acked")


# ─── agent_logs ───────────────────────────────────────────────────────────────
class AgentLog(Base):
    __tablename__ = "agent_logs"

    log_id     = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(UUID(as_uuid=True))
    agent_name = Column(String(64), nullable=False)
    event_type = Column(String(64))
    payload    = Column(JSONB)
    ts         = Column(TIMESTAMPTZ, default=datetime.utcnow)
    latency_ms = Column(Integer)
    status     = Column(String(16))  # ok | error | timeout | degraded


# ─── audit_logs ───────────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id     = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id   = Column(UUID(as_uuid=True))
    actor        = Column(String(64))   # agent name or operator_id
    action       = Column(String(128))
    before_state = Column(JSONB)
    after_state  = Column(JSONB)
    ts           = Column(TIMESTAMPTZ, default=datetime.utcnow)
