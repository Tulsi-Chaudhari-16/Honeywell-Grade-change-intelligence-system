# Honeywell Grade Change Intelligence System (GCIS)

An AI-driven, multi-agent orchestration system designed to optimize the grade change process in paper manufacturing. The system leverages real-time IoT sensor data, machine learning pipelines, and LangGraph-powered intelligent agents to reduce off-spec production and provide operators with actionable, physics-validated recommendations.

## 🌟 Key Features

* **Real-time IoT Intelligence:** Continuously monitors paper machine sensors (speed, steam pressure, etc.) and Quality Control System (QCS) data.
* **Predictive ML Pipeline:** 
  * **LightGBM:** Calculates the real-time probability of off-spec production (`p_offspec`).
  * **Temporal Convolutional Networks (TCN):** Forecasts the trajectory of the machine state.
  * **TreeExplainer (SHAP):** Performs real-time root-cause attribution to identify driving factors.
* **Agentic Orchestration:** A Node.js LangGraph `Supervisor Agent` orchestrates specialized sub-agents:
  * *Prediction Agent:* Assesses system state.
  * *Root Cause Agent:* Analyzes SHAP drivers.
  * *Historical Retrieval Agent:* Queries vector embeddings (Qdrant) for past successful interventions.
  * *Recommendation & Safety Agents:* Generates and validates physics-constrained fixes.
  * *AI Explanation Agent:* Synthesizes operator-friendly narratives using Groq LLMs.
* **Operator Dashboard:** A Next.js responsive UI featuring live WebSocket telemetry, predictive timelines, and interactive AI alerts.

## 🏗️ Architecture Stack

* **Frontend:** Next.js (React), TailwindCSS, Recharts
* **Agentic Backend:** Node.js, Express, LangGraph, Groq (llama-3.3-70b-versatile)
* **ML Service:** Python, FastAPI, LightGBM, PyTorch, SHAP
* **Infrastructure:** PostgreSQL, Redis (Pub/Sub), Qdrant (Vector DB)

## 🚀 Getting Started

### Prerequisites
* Node.js (v18+)
* Python (3.10+)
* Redis Server & MongoDB (running locally or via Docker)

### 1. Install ML Dependencies & Run API
```bash
cd backend
pip install -r requirements-ml.txt
python ml_api.py
```
*(Runs the FastAPI ML engine on port 8001)*

### 2. Install Node Backend & Run Agents
```bash
cd node-backend
npm install
npm run dev
```
*(Runs the Express/LangGraph server and WebSockets on port 8000)*

### 3. Start the Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
*(Dashboard available at http://localhost:3000)*

## 📂 Repository Structure
* `/backend` - Python FastAPI application, ML models, and inference scripts.
* `/node-backend` - Node.js LangGraph orchestration and WebSocket servers.
* `/frontend` - Next.js React application for the operator dashboard.
* `/data` - Seed data, simulation notebooks, and exploratory data analysis.

---
*Built for the Honeywell Hackathon.*