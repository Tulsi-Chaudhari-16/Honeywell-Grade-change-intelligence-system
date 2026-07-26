-- ============================================================
-- GCIS Initial Schema — Migration 001
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── recipes: grade targets ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grade_code          VARCHAR(32) NOT NULL,
    grade_name          VARCHAR(128),
    target_basis_weight NUMERIC(8,2),
    target_moisture     NUMERIC(6,3),
    target_ash          NUMERIC(6,3),
    target_caliper      NUMERIC(8,3),
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── machine_limits: hard safety limits per variable ─────────────────────────
CREATE TABLE IF NOT EXISTS machine_limits (
    limit_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id          VARCHAR(32) NOT NULL,
    variable_name       VARCHAR(64) NOT NULL,
    min_value           NUMERIC,
    max_value           NUMERIC,
    max_rate_of_change  NUMERIC,
    unit                VARCHAR(16),
    UNIQUE(machine_id, variable_name)
);

-- ─── operators ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS operators (
    operator_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name        VARCHAR(128),
    shift               VARCHAR(16),
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── transitions: one row per grade change episode ───────────────────────────
CREATE TABLE IF NOT EXISTS transitions (
    episode_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id          VARCHAR(32) NOT NULL,
    from_recipe_id      UUID REFERENCES recipes(recipe_id),
    to_recipe_id        UUID REFERENCES recipes(recipe_id),
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    outcome             VARCHAR(16),           -- 'success' | 'offspec' | 'unconfirmed'
    recovery_time_min   NUMERIC,
    final_basis_weight  NUMERIC(8,2),
    final_moisture      NUMERIC(6,3),
    final_ash           NUMERIC(6,3),
    embedding_id        VARCHAR(64),           -- pointer to Qdrant point id
    current_state       VARCHAR(32) DEFAULT 'Monitoring',
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── predictions: model output per episode step ──────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id          UUID REFERENCES transitions(episode_id),
    p_offspec           NUMERIC(5,4) NOT NULL,
    trajectory          JSONB,                 -- list[float]
    eta_stabilize_min   NUMERIC(6,2),
    confidence          NUMERIC(5,4),
    model_version       VARCHAR(64),
    degraded_mode       BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── feature_snapshots: training data accumulation ───────────────────────────
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id          UUID REFERENCES transitions(episode_id),
    feature_vector      JSONB NOT NULL,        -- serialized FeatureVector
    window_start        TIMESTAMPTZ,
    window_end          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── recommendations: proposed + validated setpoint changes ──────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id          UUID REFERENCES transitions(episode_id),
    prediction_id       UUID REFERENCES predictions(prediction_id),
    variable_name       VARCHAR(64),
    direction           VARCHAR(16),           -- 'increase' | 'decrease'
    proposed_value      NUMERIC,
    clamped_value       NUMERIC,
    confidence          NUMERIC(5,4),
    historical_support  JSONB,                 -- array of transition_ids cited
    safety_status       VARCHAR(16),           -- 'approved' | 'clamped' | 'rejected'
    rejection_reason    TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── feedback: operator Accept/Reject/Modify records ─────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id   UUID REFERENCES recommendations(recommendation_id),
    operator_id         UUID REFERENCES operators(operator_id),
    action              VARCHAR(16) NOT NULL,  -- 'accepted' | 'rejected' | 'modified'
    modified_value      NUMERIC,
    note                TEXT,
    ts                  TIMESTAMPTZ DEFAULT now()
);

-- ─── alerts ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    alert_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id          UUID REFERENCES transitions(episode_id),
    severity            VARCHAR(16) NOT NULL,  -- 'critical' | 'warning' | 'info'
    alert_type          VARCHAR(64) NOT NULL,
    message             TEXT,
    ts                  TIMESTAMPTZ DEFAULT now(),
    acknowledged_by     UUID REFERENCES operators(operator_id),
    acknowledged_at     TIMESTAMPTZ,
    dedup_key           VARCHAR(128)           -- for deduplication window
);

-- ─── agent_logs: per-agent latency and status ────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_logs (
    log_id              BIGSERIAL PRIMARY KEY,
    episode_id          UUID,
    agent_name          VARCHAR(64) NOT NULL,
    event_type          VARCHAR(64),
    payload             JSONB,
    ts                  TIMESTAMPTZ DEFAULT now(),
    latency_ms          INT,
    status              VARCHAR(16)            -- 'ok' | 'error' | 'timeout' | 'degraded'
);

-- ─── audit_logs: immutable audit trail ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id            BIGSERIAL PRIMARY KEY,
    episode_id          UUID,
    actor               VARCHAR(64),           -- agent name or operator_id
    action              VARCHAR(128),
    before_state        JSONB,
    after_state         JSONB,
    ts                  TIMESTAMPTZ DEFAULT now()
);

-- ─── Indexes for common query patterns ───────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_transitions_machine_started ON transitions(machine_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_transitions_outcome ON transitions(outcome);
CREATE INDEX IF NOT EXISTS idx_predictions_episode ON predictions(episode_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_episode ON recommendations(episode_id);
CREATE INDEX IF NOT EXISTS idx_feedback_recommendation ON feedback(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_alerts_episode ON alerts(episode_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, acknowledged_at);
CREATE INDEX IF NOT EXISTS idx_agent_logs_episode ON agent_logs(episode_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent ON agent_logs(agent_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_episode ON audit_logs(episode_id, ts DESC);
