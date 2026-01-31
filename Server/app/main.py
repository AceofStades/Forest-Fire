import os
import pickle
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Import your custom modules
from .model_wrapper import load_model 
from .d_star_lite import DStarLite
from .schemas import Features, PredictionResponse

# 1. State Management
ml_artifacts: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    # Based on your previous message, you placed the 'models' folder INSIDE 'app'
    # So the path relative to Server/ is "app/models/..."
    model_path = os.getenv("MODEL_PATH", "app/models/model.pkl")
    grid_path = os.getenv("GRID_PATH", "app/models/fire_prediction_sample.npy")
    stats_path = os.getenv("STATS_PATH", "app/models/stats_cache.pkl")
    
    try:
        # Load Base Grid (The "Memory" of the forest)
        if os.path.exists(grid_path):
            base_grid = np.load(grid_path)
            ml_artifacts["base_grid"] = base_grid.copy() # Keep original safe
            ml_artifacts["fire_grid"] = base_grid.copy() # This one gets modified
            print(f"✅ Fire Grid loaded. Shape: {base_grid.shape}")
        else:
            print(f"❌ CRITICAL: Fire Grid not found at {grid_path}")
            ml_artifacts["fire_grid"] = None

        if os.path.exists(model_path):
            ml_artifacts["model"] = load_model(model_path)
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        
    yield
    ml_artifacts.clear()
    

app = FastAPI(title="Forest Fire Rescue API", lifespan=lifespan)

# 2. CORS Configuration (FIXED: Only ONE Middleware Block)
# This allows both localhost:3000 and 127.0.0.1:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_ORDER = ["temperature", "humidity", "wind", "rain"]

# 3. Pydantic Models
class PathRequest(BaseModel):
    start: List[int] # [row, col]
    goal: List[int]  # [row, col]
    
class SimulationParams(BaseModel):
    wind_speed: float       # 0 to 100 km/h
    wind_direction: str     # "N", "S", "E", "W"
    humidity: float         # 0 to 100 %

# 4. Endpoints

@app.get("/health")
def health():
    # If grid is None, status is degraded
    status = "ok" if ml_artifacts.get("fire_grid") is not None else "degraded"
    return {"status": status}

@app.get("/fire-grid")
async def get_fire_grid():
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        # This 500 error causes the "Loading" hang on frontend if not handled
        raise HTTPException(status_code=500, detail="Fire grid not loaded on server")
    return {"grid": grid.tolist()}
@app.post("/update-simulation")
async def update_simulation(params: SimulationParams):
    """Updates the fire grid based on physical factors"""
    base = ml_artifacts.get("base_grid")
    if base is None:
        raise HTTPException(status_code=500, detail="Base grid not loaded")
    
    # 1. Start with the original probability
    new_grid = base.copy()
    
    # 2. Apply Wind Factor
    # Wind pushes fire. If Wind is "North" (blowing South),
    # cells to the SOUTH of a fire risk get higher probability.
    rows, cols = new_grid.shape
    wind_factor = (params.wind_speed / 100.0) * 0.3 # Max 30% increase
    
    if params.wind_direction == "N": # Blowing South
        new_grid[1:, :] += (new_grid[:-1, :] * wind_factor)
    elif params.wind_direction == "S": # Blowing North
        new_grid[:-1, :] += (new_grid[1:, :] * wind_factor)
    elif params.wind_direction == "E": # Blowing West
        new_grid[:, :-1] += (new_grid[:, 1:] * wind_factor)
    elif params.wind_direction == "W": # Blowing East
        new_grid[:, 1:] += (new_grid[:, :-1] * wind_factor)

    # 3. Apply Humidity Factor
    # High humidity dampens fire spread
    humidity_factor = (params.humidity / 100.0) * 0.5 # Max 50% reduction
    new_grid = new_grid * (1.0 - humidity_factor)
    
    # 4. Clamp and Save
    np.clip(new_grid, 0, 1, out=new_grid)
    ml_artifacts["fire_grid"] = new_grid
    
    return {"status": "updated", "max_risk": float(np.max(new_grid))}


@app.post("/get-safe-path")
async def get_safe_path(req: PathRequest):
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        raise HTTPException(status_code=500, detail="Fire data not loaded")
    
    start_tuple = tuple(req.start)
    goal_tuple = tuple(req.goal)
    
    # Run D* Lite
    planner = DStarLite(grid, start_tuple, goal_tuple)
    planner.compute_shortest_path()
    
    # Reconstruct Path
    path = [list(start_tuple)]
    curr = start_tuple
    steps = 0
    
    while curr != goal_tuple and steps < 3000:
        neighbors = planner.get_neighbors(curr)
        # Greedy descent
        if not neighbors: break
        curr = min(neighbors, key=lambda n: planner.g.get(n, float('inf')))
        path.append(list(curr))
        steps += 1
            
    return {"path": path}

# Predict endpoint...
@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features):
    model = ml_artifacts.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    X = [[getattr(features, f) for f in FEATURE_ORDER]]
    pred = model.predict(X)[0]
    prob = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None
    return {"prediction": int(pred), "probability": prob}