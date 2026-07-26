# GCIS — Partner Part: Agents, Backend & Frontend Scope

> **Role**: Agent Orchestration, Backend Engineering, Frontend  
> **I handle**: ML models and feature engineering (see `my-part-ml-models.md`)

---

## 0. What You Are Building

Everything except the ML models themselves:
- The **13-agent LangGraph system** that orchestrates the decision pipeline
- The **FastAPI backend** — ingestion, APIs, WebSockets, Kafka, Celery
- The **Next.js frontend** — 15 dashboard pages, real-time UI, operator controls
- The **infrastructure** — Docker Compose, Postgres, Redis, Qdrant, Kafka

The ML numbers (p_offspec, trajectory, SHAP values, embeddings) come from my side. Your job is to wire them into the agent pipeline, present them in the UI, and build all the operator interaction flows.

---

## 1. Multi-Agent System Overview (Section 2)

**Framework**: LangGraph (not CrewAI / AutoGen)  
**LLM**: Claude backs all reasoning agents  
**Key design rule**: Deterministic agents never use LLMs. LLM-backed agents never invent numbers — they only narrate, retrieve, and reason over numbers already computed.

### Agent Split

| Agent | Type | Your Responsibility |
|---|---|---|
| Supervisor | Deterministic (LangGraph state machine) | Owns episode state machine, routes all agents |
| Data Agent | Deterministic | Kafka tag ingestion, validation, Historian backfill |
| Feature Engineering Agent | Deterministic | Calls ML feature.py, publishes FeatureVectors |
| Prediction Agent | Deterministic (ML inference) | Calls my model inference wrappers |
| Root Cause Agent | LLM-backed | Runs SHAP (from my output), narrates with Claude |
| Recommendation Agent | LLM-backed | Proposes setpoint changes grounded in RCA + history |
| Safety Validation Agent | Deterministic (rules-only) | Hard gate — no recommendation ships without this |
| Historical Retrieval Agent | LLM-backed (narration only) | Embeds current episode, queries Qdrant, narrates matches |
| Explanation Agent | LLM-backed | Synthesises final operator-facing explanation card |
| Learning Agent | Mostly deterministic | Closes episode loop, indexes to Qdrant, triggers retraining |
| Alert Agent | Deterministic | Fires on p_offspec threshold, sensor staleness, agent failures |
| Operator Feedback Agent | Deterministic | Captures Accept/Reject/Modify from UI |
| Dashboard Agent | Deterministic | Aggregates all agent outputs for WebSocket push |

---

## 2. Episode State Machine (Section 3 & 8)

Each grade transition is one **episode** — a LangGraph graph with these states:

```
Monitoring
  --> FeatureBuilding (new tag data)
  --> Predicting
      --> Monitoring (p_offspec below threshold)
      --> RootCause (p_offspec above threshold)
          --> Retrieving
          --> Recommending
          --> Validating
              --> Rejected (fails safety rules) --> Monitoring
              --> Explaining (passes / clamped)
                  --> AwaitingOperator
                      --> Applied (operator accepts) --> Monitoring
                      --> Dismissed (operator rejects) --> Monitoring
                      --> Modified (operator modifies) --> Monitoring
  --> EpisodeClosed (transition ends)
      --> Learning --> [end]
```

### Guard Conditions (LangGraph conditional edges)
- `Monitoring --> RootCause` only if `p_offspec >= ALERT_THRESHOLD (default 0.65)` **AND** `confidence >= MIN_CONFIDENCE (default 0.4)`
- `Validating --> Rejected` if ANY candidate fails safety rules with no valid clamp
- `AwaitingOperator` has a **3-minute timeout** — auto-transitions to `Dismissed` with reason `timed_out`

---

## 3. Agent Contracts (Section 2 details)

### 3.1 Supervisor Agent
- Deterministic state machine (LangGraph graph)
- Owns master audit log entry per episode
- **Failure**: If Safety Validation Agent fails → no recommendation ever shown (fail closed, hard rule)
- **File**: `backend/app/agents/supervisor_graph.py`

### 3.2 Data Agent
- Subscribes to Kafka tag streams, validates schema/units
- Handles sensor dropout / stale-tag detection, backfills from Historian
- Stale tag > threshold (10s fast / 60s slow) → emits `SensorStaleEvent` to Alert Agent
- **Memory**: Last-known-good value per tag in Redis
- **File**: `backend/app/agents/data_agent.py`

### 3.3 Feature Engineering Agent
- Builds sliding-window features by calling `ml/features.py` (my code)
- Publishes `FeatureVector` to Redis stream `stream:feature_vectors`
- Stores snapshots to Postgres `feature_snapshots` for training data accumulation
- **File**: `backend/app/agents/feature_engineering_agent.py`

### 3.4 Prediction Agent
- Calls my inference wrappers from `ml/infer/`
- Receives `Prediction {p_offspec, trajectory, eta_stabilize_min, confidence, model_version, degraded_mode}`
- Publishes to Root Cause Agent and Alert Agent
- **Degraded mode**: If model unavailable, serves last-deployed-good model; if none, falls back to Isolation Forest + rule-based check
- **File**: `backend/app/agents/prediction_agent.py`

### 3.5 Root Cause Agent
- Receives `Prediction` + my SHAP output (`RootCauseAttribution`)
- LLM step (Claude) narrates SHAP values — it does NOT compute them
- **SHAP fallback**: If timeout → use precomputed permutation importance, flagged as `attribution_method=fallback`
- **File**: `backend/app/agents/root_cause_agent.py`

**Claude prompt**:
```
You are a paper-mill process engineer. You are given a ranked list of
SHAP feature attributions for a predicted Basis Weight excursion, plus
the current values and rates of change of the top factors, plus how far
each deviates from this grade transition's historical successful envelope.
Do not invent any numeric value not present in the input. Write a 2-3
sentence causal explanation in the style of an experienced operator's
shift note. Name the top 2 factors only. State direction (too high/too
low/too fast) and compare to the historical envelope in relative terms
only (e.g. "faster than typical successful transitions"), never inventing
a percentage that isn't in the input data.
```

### 3.6 Recommendation Agent
- Candidates ONLY drawn from: (a) variables Root Cause Agent flagged AND (b) actions observed in historical matches — LLM cannot propose anything outside these lists
- Optional constrained optimizer (scipy minimize with recipe/machine limit bounds) generates numeric candidates; LLM selects and narrates
- Sends candidates to Safety Validation Agent before anything reaches an operator
- **File**: `backend/app/agents/recommendation_agent.py`

**Claude prompt**:
```
You are proposing corrective setpoint changes for a paper machine grade
transition. You are given: the top contributing factors to a predicted
off-spec event, the 3 most similar historical transitions and what
operators changed in each, and the hard recipe/machine limits for this
grade. Propose at most 3 candidate actions, each expressed as a variable,
a direction, and a bounded magnitude (percentage or absolute unit) that
a human operator could apply. Every magnitude you propose MUST fall
strictly within the provided recipe/machine limits — if you cannot justify
a magnitude from the historical matches or the root-cause factors, do not
propose one. For each candidate, state which historical transition(s)
support it and the expected improvement observed in that history. Do not
fabricate a historical match that wasn't provided to you.
```

### 3.7 Safety Validation Agent — CRITICAL GATE
- **Deterministic, rule-based, no LLM** — safety logic must be auditable and testable
- Validates every candidate against: recipe limits, machine hard limits, rate-of-change limits, interaction rules (e.g., "do not increase speed AND steam simultaneously beyond X")
- Reads limits fresh from Postgres each call (never cached beyond short TTL)
- Any rule-evaluation error → reject entirely (fail closed, non-configurable)
- Rejected recommendations: logged to `audit_logs`, shown only in debug panel — never as actionable
- **File**: `backend/app/agents/safety_validation_agent.py`

### 3.8 Historical Retrieval Agent
- Embeds current episode FeatureVector (using my embedding model)
- Qdrant ANN search: `top_k=5`, filtered by `grade_pair` BEFORE similarity ranking
- Fetches top 3 structured records (what operators changed, success, recovery time)
- **Fallback**: Qdrant unavailable → structured SQL Euclidean-distance query on Postgres
- **File**: `backend/app/agents/historical_retrieval_agent.py`

### 3.9 Explanation Agent
- Combines: RootCauseReport + ValidatedRecommendation + HistoricalMatches
- Outputs `ExplanationCard` matching exact UI schema (Section 10)
- Any missing upstream field → show "Not available" — never fabricated filler
- **File**: `backend/app/agents/explanation_agent.py`

**ExplanationCard fields** (7 mandatory fields, always):
`prediction_summary` · `reason` · `evidence` · `historical_match` · `confidence_statement` · `expected_improvement` · `safety_check_status`

### 3.10 Learning Agent
- On episode close: writes outcome + operator decisions to vector store (Qdrant), training dataset, and drift metrics
- If lab-result ground truth never arrives → archive as `outcome=unconfirmed`, excluded from supervised retraining but indexed in Qdrant with `predicted` flag
- Emits `RetrainRequested` event to offline pipeline (not to any live agent)
- **File**: `backend/app/agents/learning_agent.py`

### 3.11 Alert Agent
- Fires on: `p_offspec` threshold crossed, sensor staleness, model degradation, agent failures
- De-duplication window (don't re-fire same alert every second)
- **No external dependencies** beyond Redis — must never fail silently
- **File**: `backend/app/agents/alert_agent.py`

### 3.12 Operator Feedback Agent
- Captures Accept / Reject / Modify + optional free-text notes from UI
- Feedback write failure → exponential backoff retry; UI shows optimistic confirmation
- **File**: `backend/app/agents/operator_feedback_agent.py`

### 3.13 Dashboard Agent
- Aggregates all agent outputs into the live state object for WebSocket push
- Serves last-known-good state with `stale: true` flag on any upstream agent failure
- **File**: `backend/app/agents/dashboard_agent.py`

---

## 4. Backend Architecture (Section 9)

| Component | Choice | Notes |
|---|---|---|
| API framework | FastAPI | Async, WebSocket-native, auto OpenAPI docs |
| Streaming ingestion | Kafka (KRaft mode, no Zookeeper) | Decouples tag ingestion from processing; replay for backtesting |
| Hot state / pub-sub | Redis | Sub-100ms rolling buffers; LangGraph checkpoint backend |
| Task queue | Celery (Redis broker) | Async dispatch for LLM-backed reasoning agents |
| Relational store | PostgreSQL | ACID, audit trail |
| Vector store | Qdrant | Self-hosted Docker, metadata-filter-before-ANN |
| Realtime push | WebSockets (FastAPI native) | Sub-second dashboard updates |
| Containerization | Docker Compose | One-command reproducible demo |

---

## 5. REST API Contracts (Section 16)

```
GET  /api/v1/episodes/active
GET  /api/v1/episodes/{episode_id}
GET  /api/v1/episodes/{episode_id}/predictions
GET  /api/v1/episodes/{episode_id}/recommendations
POST /api/v1/recommendations/{recommendation_id}/feedback
       body: { action: "accepted"|"rejected"|"modified", modified_value?, note? }
GET  /api/v1/episodes/{episode_id}/historical-matches
GET  /api/v1/episodes/{episode_id}/root-cause
GET  /api/v1/recipes
GET  /api/v1/recipes/{recipe_id}/limits
GET  /api/v1/alerts?severity=&acknowledged=
POST /api/v1/alerts/{alert_id}/acknowledge
GET  /api/v1/system/health
GET  /api/v1/system/drift
POST /api/v1/simulate
       body: { episode_id, overrides: { variable_name: value } }
       returns: { trajectory[], p_offspec, safety_flags[] }
WS   /ws/episodes/{episode_id}      # live dashboard state stream
WS   /ws/alerts                     # global alert stream
```

---

## 6. DB Tables You Own

```sql
-- recipes: grade targets
CREATE TABLE recipes (
    recipe_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grade_code      VARCHAR(32) NOT NULL,
    grade_name      VARCHAR(128),
    target_basis_weight NUMERIC(8,2),
    target_moisture NUMERIC(6,3),
    target_ash      NUMERIC(6,3),
    target_caliper  NUMERIC(8,3),
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- machine_limits: hard safety limits per variable
CREATE TABLE machine_limits (
    limit_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id      VARCHAR(32) NOT NULL,
    variable_name   VARCHAR(64) NOT NULL,
    min_value       NUMERIC,
    max_value       NUMERIC,
    max_rate_of_change NUMERIC,
    unit            VARCHAR(16),
    UNIQUE(machine_id, variable_name)
);

-- operators
CREATE TABLE operators (
    operator_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name    VARCHAR(128),
    shift           VARCHAR(16),
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- transitions: one row per grade change episode
CREATE TABLE transitions (
    episode_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id      VARCHAR(32) NOT NULL,
    from_recipe_id  UUID REFERENCES recipes(recipe_id),
    to_recipe_id    UUID REFERENCES recipes(recipe_id),
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    outcome         VARCHAR(16),  -- 'success' | 'offspec' | 'unconfirmed'
    recovery_time_min NUMERIC,
    final_basis_weight NUMERIC(8,2),
    final_moisture  NUMERIC(6,3),
    final_ash       NUMERIC(6,3),
    embedding_id    VARCHAR(64)   -- pointer to Qdrant point id
);

-- recommendations: proposed + validated setpoint changes
CREATE TABLE recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id      UUID REFERENCES transitions(episode_id),
    prediction_id   UUID REFERENCES predictions(prediction_id),
    variable_name   VARCHAR(64),
    direction       VARCHAR(16),
    proposed_value  NUMERIC,
    clamped_value   NUMERIC,
    confidence      NUMERIC(5,4),
    historical_support JSONB,  -- array of transition_ids cited
    safety_status   VARCHAR(16),  -- 'approved' | 'clamped' | 'rejected'
    rejection_reason TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- feedback: operator Accept/Reject/Modify records
CREATE TABLE feedback (
    feedback_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID REFERENCES recommendations(recommendation_id),
    operator_id     UUID REFERENCES operators(operator_id),
    action          VARCHAR(16),  -- 'accepted' | 'rejected' | 'modified'
    modified_value  NUMERIC,
    note            TEXT,
    ts              TIMESTAMPTZ DEFAULT now()
);

-- alerts
CREATE TABLE alerts (
    alert_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id      UUID REFERENCES transitions(episode_id),
    severity        VARCHAR(16),
    alert_type      VARCHAR(64),
    message         TEXT,
    ts              TIMESTAMPTZ DEFAULT now(),
    acknowledged_by UUID REFERENCES operators(operator_id),
    acknowledged_at TIMESTAMPTZ
);

-- agent_logs: per-agent latency and status
CREATE TABLE agent_logs (
    log_id          BIGSERIAL PRIMARY KEY,
    episode_id      UUID,
    agent_name      VARCHAR(64),
    event_type      VARCHAR(64),
    payload         JSONB,
    ts              TIMESTAMPTZ DEFAULT now(),
    latency_ms      INT,
    status          VARCHAR(16)  -- 'ok' | 'error' | 'timeout' | 'degraded'
);

-- audit_logs: immutable audit trail
CREATE TABLE audit_logs (
    audit_id        BIGSERIAL PRIMARY KEY,
    episode_id      UUID,
    actor           VARCHAR(64),  -- agent name or operator_id
    action          VARCHAR(128),
    before_state    JSONB,
    after_state     JSONB,
    ts              TIMESTAMPTZ DEFAULT now()
);
```

---

## 7. Frontend — Next.js Dashboard (Section 10)

### Stack
- **Next.js** (App Router) + React + Tailwind + shadcn/ui
- **Framer Motion** — state-transition animations, alert banners, gauge animations
- **Recharts** — standard charts (prediction timeline, trend graphs)
- **D3** — SHAP waterfall, correlation matrix heatmap, Sankey, parallel coordinates
- **Theme**: Dark — near-black slate background + amber accent. Monospace font for all numeric tag values.

### 15 Pages

| # | Page | Key Content |
|---|---|---|
| 1 | **Overview** | Mill-wide status tiles, live sparkline strip per machine, open alerts |
| 2 | **Live Process** | Real-time tag values grouped by subsystem, recipe-limit bands overlaid |
| 3 | **Prediction Timeline** | Time-series chart: predicted vs actual BW with ±2.5% band, forecast cone, recommendation markers |
| 4 | **Historical Similarity Explorer** | Top-K retrieved transitions as cards, trajectory comparison overlay |
| 5 | **Root Cause** | SHAP-style waterfall, ranked factor list, each factor clickable for trend line, radar vs historical envelope |
| 6 | **Recommendation Panel** | Accept/Reject/Modify action card, full ExplanationCard fields, safety-check badge |
| 7 | **Correlation Explorer** | D3 cross-variable correlation matrix heatmap, filterable by grade pair + time range |
| 8 | **Feature Importance** | Global (model-level SHAP) vs local (this-episode) toggle |
| 9 | **Transition Timeline** | Sankey/Gantt across all machines today, colored by outcome |
| 10 | **Recipe Explorer** | Browse recipes and limits, historical transitions per recipe |
| 11 | **Operator Feedback** | Accept/reject/modify history, filterable by operator/shift |
| 12 | **AI Explanation (debug)** | Rejected candidates + raw agent-log trace for technical judges |
| 13 | **Reports** | Auto-generated shift/episode reports (LLM-summarized from structured data) |
| 14 | **Alerts** | Alert feed with severity filtering and acknowledgment workflow |
| 15 | **System Health** | Agent latency/error/status, Kafka lag, model version, drift metric trend |

### Component Hierarchy (top level)
```
App
 ├─ DashboardShell (nav, alert banner, connection status)
 │   ├─ Sidebar (page nav)
 │   └─ PageOutlet
 │       ├─ OverviewPage --> MachineStatusTile[]
 │       ├─ LiveProcessPage --> TagGroupPanel[] --> TagGauge / TagTrendMini
 │       ├─ PredictionTimelinePage --> PredictionChart (custom uncertainty band)
 │       ├─ HistoricalSimilarityPage --> TransitionMatchCard[] --> TrajectoryOverlayModal
 │       ├─ RootCausePage --> ShapWaterfallChart, FactorTrendDrawer
 │       ├─ RecommendationPanel --> ExplanationCard, AcceptRejectModifyControls
 │       ├─ CorrelationExplorerPage --> CorrelationHeatmap (D3)
 │       └─ ...
 └─ WebSocketProvider (context: live episode state)
```

---

## 8. Visualization Catalogue (Section 11)

| Chart | Library | Page |
|---|---|---|
| Prediction timeline + uncertainty cone | Recharts + custom SVG overlay | Prediction Timeline |
| SHAP waterfall | D3 (custom) | Root Cause, Feature Importance |
| Correlation matrix heatmap | D3 | Correlation Explorer |
| Sankey (transition flow across grades/outcomes) | D3-sankey | Transition Timeline |
| Parallel coordinates (multi-variable transition comparison) | D3 | Historical Similarity Explorer |
| Radar chart (current vs. historical-envelope shape) | Recharts | Root Cause |
| Gauge / confidence meter | Custom SVG + Framer Motion | Recommendation Panel, Overview |
| Trend graphs (per-tag) | Recharts | Live Process |
| Transition replay (scrubber over historical episode) | Custom + Framer Motion | Historical Similarity Explorer |
| Digital twin animation (simplified schematic) | Custom SVG/Canvas | Overview (stretch goal) |

---

## 9. Digital Twin / What-If Simulation (Section 13)

- User proposes a hypothetical setpoint change via UI slider
- Backend re-runs Feature Engineering + Prediction Agent pipeline with that feature overridden
- Returns new forecast trajectory overlaid on real one
- Safety Validation Agent runs on simulated proposals too
- **Scope it explicitly as ML-surrogate simulation, not first-principles physics twin**

---

## 10. Hackathon "Wow" Features (Section 15, prioritized)

1. **Chat with the Factory** — natural-language query over current + historical episode data, backed by same LangGraph tools as Recommendation Agent (reuse, don't rebuild) — highest judge impact per build effort
2. **Transition Replay** — scrub through a past off-spec event with prediction/root-cause/recommendation timeline replaying in sync
3. **Auto shift report generation** — LLM summarizes the day's episodes from structured data (same anti-hallucination pattern)
4. **Digital twin what-if slider** (see above)
5. **Voice-AI interface** — cut first if time is short

---

## 11. Agentic AI Implementation Detail (Section 12)

- **LangGraph** for episode graph; **Claude** for all LLM reasoning nodes
- Each reasoning node gets a **constrained toolset** — e.g., Recommendation Agent can only call `get_recipe_limits`, `get_machine_limits`, `get_root_cause_report`, `get_historical_matches` — it cannot run raw SQL or vector search directly (those are the Historical Retrieval Agent's tools only)
- **No shared long-term conversational memory** across nodes — each node receives exactly the structured inputs the graph edges provide. Long-term memory = vector store (Qdrant) + Postgres
- **MCP**: exposes SQL + vector-search tools to LangGraph tool-calling nodes in a swappable way

---

## 12. Your File Structure

```
backend/app/
├── main.py
├── api/
│   ├── episodes.py
│   ├── recommendations.py
│   ├── recipes.py
│   ├── alerts.py
│   ├── simulate.py
│   └── websocket.py
├── agents/
│   ├── supervisor_graph.py          # LangGraph graph definition
│   ├── data_agent.py
│   ├── feature_engineering_agent.py
│   ├── prediction_agent.py
│   ├── root_cause_agent.py
│   ├── recommendation_agent.py
│   ├── safety_validation_agent.py
│   ├── historical_retrieval_agent.py
│   ├── explanation_agent.py
│   ├── learning_agent.py
│   ├── alert_agent.py
│   ├── operator_feedback_agent.py
│   └── dashboard_agent.py
├── db/
│   ├── models.py
│   └── session.py
├── vectorstore/
│   └── qdrant_client.py
└── core/
    ├── config.py
    └── redis_client.py

frontend/
├── app/
│   ├── overview/
│   ├── live-process/
│   ├── prediction-timeline/
│   ├── historical-explorer/
│   ├── root-cause/
│   ├── recommendations/
│   ├── correlation-explorer/
│   ├── feature-importance/
│   ├── transition-timeline/
│   ├── recipes/
│   ├── feedback/
│   ├── ai-explanation/
│   ├── reports/
│   ├── alerts/
│   └── system-health/
├── components/
└── lib/
    └── ws-client.ts

infra/
├── kafka/
├── postgres/
│   └── migrations/
└── qdrant/

docker-compose.yml
```

---

## 13. Interface Contract with Me (What You Consume)

### From my Prediction Agent wrapper
```python
Prediction:
  p_offspec: float          # [0, 1] — triggers alert at >= 0.65
  trajectory: list[float]   # N-step ahead BW forecast
  eta_stabilize_min: float  # estimated minutes to stabilize
  confidence: float         # blended model + historical support confidence
  model_version: str
  degraded_mode: bool       # True if falling back to rule-based
```

### From my SHAP output (Root Cause Agent consumes this)
```python
RootCauseAttribution:
  ranked_factors: list[{
    feature_name: str,
    shap_value: float,
    feature_value: float,
    direction: str,           # 'too_high' | 'too_low' | 'too_fast'
    rank: int
  }]
  attribution_method: str     # 'shap' | 'permutation_fallback'
```

### From my embedding model (Historical Retrieval Agent consumes this)
- A fixed-length float vector per episode window, callable as `ml/infer/embed_transition(feature_vector_sequence) -> np.ndarray`

---

## 14. Phase 0 Hackathon Build Targets

- [ ] Docker Compose setup (Kafka KRaft, Postgres, Redis, Qdrant, FastAPI, Next.js)
- [ ] Postgres migrations (full schema from Section 7 of full spec)
- [ ] Kafka topic setup + Data Agent (tag ingestion + validation)
- [ ] Feature Engineering Agent (calls my features.py)
- [ ] Prediction Agent (calls my inference wrappers)
- [ ] Root Cause Agent (receives my SHAP output, narrates with Claude)
- [ ] Historical Retrieval Agent (Qdrant query, narrates matches)
- [ ] Recommendation Agent (grounded candidate generation, Claude)
- [ ] Safety Validation Agent (deterministic rule engine -- CRITICAL, fail closed)
- [ ] Explanation Agent (synthesises ExplanationCard)
- [ ] Alert Agent (p_offspec threshold, sensor staleness)
- [ ] Operator Feedback Agent (Accept/Reject/Modify capture)
- [ ] Supervisor LangGraph graph (episode state machine)
- [ ] Dashboard Agent + WebSocket gateway
- [ ] REST API endpoints (Section 16 of full spec)
- [ ] 5 core frontend pages: Overview, Prediction Timeline, Root Cause, Recommendation Panel, Historical Similarity Explorer
- [ ] Learning Agent (episode close + Qdrant indexing)
