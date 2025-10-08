# Server/app/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class Features(BaseModel):
    # change/add features to match your training features
    temperature: float
    humidity: float
    wind: float
    rain: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[int]
