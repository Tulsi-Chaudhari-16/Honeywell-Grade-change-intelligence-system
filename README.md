# Honeywell GCIS (Grade Change Intelligence System)

This repo contains our submission for the Honeywell Hackathon. The goal of this project is to optimize the grade change process in paper manufacturing using machine learning and LangGraph agents to reduce off-spec production.

## How it works

We built a 3-part system:
1. **Python ML Engine (`/backend`)**: Handles feature engineering and runs our models (LightGBM for anomaly detection, TCN for forecasting, and SHAP for root cause analysis) as a FastAPI service.
2. **Node.js Orchestrator (`/node-backend`)**: Uses LangGraph to route data between different specialized agents (Prediction, Root Cause, Recommendation, etc.) and streams updates via WebSockets.
3. **Next.js Dashboard (`/frontend`)**: A real-time operator UI built with React and Tailwind to visualize the sensor data and AI recommendations.

## Local Setup

You'll need Node.js (v18+) and Python (3.10+) installed. You also need a local MongoDB and Redis server running.

### 1. Start the ML API
```bash
cd backend
pip install -r requirements-ml.txt
python ml_api.py
```
This runs the Python API on port 8001.

### 2. Start the Node Backend
```bash
cd node-backend
npm install
npm run dev
```
This starts the Express server and LangGraph agents on port 8000.

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at http://localhost:3000.

## Project Structure

- `backend/` - Python ML models and FastAPI server
- `node-backend/` - LangGraph agents and WebSocket server
- `frontend/` - Next.js React frontend
- `data/` - Datasets and Jupyter notebooks for EDA