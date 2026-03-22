import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .d_star_lite import DStarLite
from .model_wrapper import get_event_data, load_model, run_inference
from .schemas import BatchPredictionResponse, Features, PredictionResponse

# 1. State Management with Lifespan
ml_artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_artifacts["model"] = load_model()
        print("✅ Startup Complete: Model loaded.")

        if ml_artifacts["model"] is not None:
            # Pre-cache one grid for backward compatibility
            ml_artifacts["fire_grid"] = run_inference(ml_artifacts["model"])
            print(
                f"✅ Startup Complete: Grid {ml_artifacts['fire_grid'].shape} generated."
            )
        else:
            print("❌ Startup Warning: No model available.")
            ml_artifacts["fire_grid"] = None
    except Exception as e:
        print(f"❌ Startup Failed: {e}")
        ml_artifacts["model"] = None
        ml_artifacts["fire_grid"] = None

    yield
    ml_artifacts.clear()


app = FastAPI(title="Forest Fire Rescue API", lifespan=lifespan)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Schemas
class PathRequest(BaseModel):
    start: List[int]  # [row, col]
    goal: List[int]  # [row, col]


# 4. Endpoints


@app.get("/historical-events")
def get_historical_events():
    # Returning significant indices we found earlier
    return [
        {"id": 271, "name": "Event Alpha - April"},
        {"id": 559, "name": "Event Bravo - May"},
        {"id": 592, "name": "Event Charlie - Mid May"},
        {"id": 607, "name": "Event Delta - Late May"},
    ]


@app.get("/event-data/{idx}")
def fetch_event_data(idx: int, hours: int = 48):
    if ml_artifacts.get("model") is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = get_event_data(ml_artifacts["model"], idx, hours)
    data["bounds"] = [[28.71806, 77.50902], [31.49096, 81.08195]]
    return data


@app.get("/fire-grid")
async def get_fire_grid():
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        raise HTTPException(status_code=500, detail="Fire grid not loaded")
    return {"grid": grid.tolist()}


@app.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
):
    # In a real scenario, you'd parse the NetCDF or TIFF and run inference
    # For now, we simulate processing and prepare the ML artifacts
    return {"message": f"Successfully processed {file.filename}. Ready for simulation."}


@app.post("/get-safe-path")
async def get_safe_path(req: PathRequest):
    grid = ml_artifacts.get("fire_grid")
    if grid is None:
        raise HTTPException(status_code=500, detail="Fire data not loaded")

    start_tuple = tuple(req.start)
    goal_tuple = tuple(req.goal)

    planner = DStarLite(grid, start_tuple, goal_tuple)
    planner.compute_shortest_path()

    path = [list(start_tuple)]
    curr = start_tuple

    while curr != goal_tuple:
        neighbors = planner.get_neighbors(curr)
        curr = min(neighbors, key=lambda n: planner.g.get(n, float("inf")))
        path.append(list(curr))
        if len(path) > 2000:
            break

    return {"path": path}
