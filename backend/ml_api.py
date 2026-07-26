import uvicorn
from fastapi import FastAPI, HTTPException
from app.ml.types import FeatureVector
from app.ml.infer.predict import run_prediction

app = FastAPI(title="GCIS ML Inference API")

@app.post("/predict")
async def predict_endpoint(payload: dict):
    try:
        # Reconstruct FeatureVector
        fv = FeatureVector(
            episode_id=payload.get("episode_id", "unknown"),
            ts=payload.get("ts", "2026-07-26T00:00:00Z"),
            features=payload.get("features", {}),
            imputation_flags=payload.get("imputation_flags", {})
        )
        
        # historical_support_score default 0.5 if not provided
        support_score = payload.get("historical_support_score", 0.5)
        
        prediction, root_cause = run_prediction(fv, historical_support_score=support_score)
        
        return {
            "prediction": prediction.model_dump() if prediction else None,
            "root_cause_attribution": root_cause.model_dump() if root_cause else None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
