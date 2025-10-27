# Server/app/main.py
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from typing import List
from .schemas import Features, PredictionResponse, BatchPredictionResponse
from .model_wrapper import load_model

app = FastAPI(title="Forest Fire Prediction API")

# CORS: allow Next.js (adjust origins as needed)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000") or "http://localhost:3000"
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Will be replaced at startup
model = None
FEATURE_ORDER = ["temperature", "humidity", "wind", "rain"]  # change if needed

@app.on_event("startup")
def on_startup():
    global model
    model_path = os.getenv("MODEL_PATH", "/models/model.pkl")  # docker-friendly default
    try:
        model = load_model(model_path)
        print("Model loaded from", model_path)
    except Exception as e:
        model = None
        print("Failed to load model:", e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# testing
@app.post("/test", response_model=PredictionResponse)
def test(features: Features):
    return predict(features)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    # Ensure ordering of features same as training
    X = [[getattr(features, f) for f in FEATURE_ORDER]]
    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        prob = float(max(proba))
    return {"prediction": int(pred), "probability": prob}

@app.post("/predict_csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.read_csv(file.file)
    # Ensure df has the columns expected by model and in right order:
    X = df[FEATURE_ORDER]
    preds = model.predict(X)
    return {"predictions": [int(p) for p in preds]}
