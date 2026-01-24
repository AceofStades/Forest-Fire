import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from .schemas import Features, PredictionResponse, BatchPredictionResponse
from .model_wrapper import load_model
from .d_star_lite import DStarLite
from pydantic import BaseModel

# 1. State Management with Lifespan
ml_artifacts: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
   # Change these lines in your lifespan or startup function
    model_path = os.getenv("MODEL_PATH", "app/models/model.pkl")
    grid_path = os.getenv("GRID_PATH", "app/models/fire_prediction_sample.npy")
    stats_path = os.getenv("STATS_PATH", "app/models/stats_cache.pkl")
    
    try:
        ml_artifacts["model"] = load_model(model_path)
        ml_artifacts["fire_grid"] = np.load(grid_path)
        with open(stats_path, 'rb') as f:
            ml_artifacts["stats"] = pickle.load(f)
        print(f"✅ Startup Complete: Grid {ml_artifacts['fire_grid'].shape} loaded.")
    except Exception as e:
        print(f"❌ Startup Failed: {e}")
        ml_artifacts["model"] = None
        ml_artifacts["fire_grid"] = None
        
    yield
    ml_artifacts.clear()

app = FastAPI(title="Forest Fire Rescue API", lifespan=lifespan)

# 2. CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Specifically allow your Next.js port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_ORDER = ["temperature", "humidity", "wind", "rain"]

# 3. Schemas
class PathRequest(BaseModel):
    start: List[int] # [row, col]
    goal: List[int]  # [row, col]

# 4. Endpoints
@app.get("/health")
def health():
    return {"status": "ok" if ml_artifacts.get("fire_grid") is not None else "degraded"}

@app.get("/fire-grid")
async def get_fire_grid():
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        raise HTTPException(status_code=500, detail="Fire grid not loaded")
    return {"grid": grid.tolist()}

@app.post("/get-safe-path")
async def get_safe_path(req: PathRequest):
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        raise HTTPException(status_code=500, detail="Fire data not loaded")
    
    # D* Lite needs tuples for dictionary keys
    start_tuple = tuple(req.start)
    goal_tuple = tuple(req.goal)
    
    planner = DStarLite(grid, start_tuple, goal_tuple)
    planner.compute_shortest_path()
    
    # Path construction: follow the lowest g-values
    path = [list(start_tuple)]
    curr = start_tuple
    
    while curr != goal_tuple:
        neighbors = planner.get_neighbors(curr)
        # Find neighbor with smallest cost to goal
        curr = min(neighbors, key=lambda n: planner.g.get(n, float('inf')))
        path.append(list(curr))
        
        if len(path) > 2000: # Increased safety break for 320x400 grid
            break
            
    return {"path": path}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features):
    model = ml_artifacts.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = [[getattr(features, f) for f in FEATURE_ORDER]]
    pred = model.predict(X)[0]
    prob = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None
    return {"prediction": int(pred), "probability": prob}
