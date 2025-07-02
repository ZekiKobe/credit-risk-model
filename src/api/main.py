from fastapi import FastAPI
from pydantic_models import PredictionInput, PredictionOutput
import mlflow
import pandas as pd
import numpy as np
import logging
from typing import List

app = FastAPI(title="Credit Risk API", version="1.0.0")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load model from MLflow
MODEL_URI = "models:/credit_risk_best_model/production"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info(f"Loaded model from {MODEL_URI}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """Make risk predictions for customer transactions"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Preprocess and predict
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[:,1][0]
        
        # Determine risk category
        category = "high" if proba >= 0.5 else "low"
        
        return {
            "customer_id": data.AccountId,
            "risk_probability": float(proba),
            "risk_category": category,
            "model_version": model.metadata.run_id
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}