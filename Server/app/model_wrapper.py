# Server/app/model_wrapper.py
import os
import joblib
from typing import Any

MODEL_ENV = "MODEL_PATH"

def load_model(model_path: str | None = None) -> Any:
    """
    Load and return your trained model. Adjust if your model is not a sklearn-like object.
    """
    path = model_path or os.getenv(MODEL_ENV) or "../Model/model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    return model
