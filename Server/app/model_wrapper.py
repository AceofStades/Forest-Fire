import os
import sys
from typing import Any

import numpy as np
import torch

# Ensure we can import from Model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Model"))
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

import torch.nn.functional as F
from src.dataset import CACHE_PATH, _compute_global_stats, _load_ds
from src.models import UNet

MODEL_ENV = "MODEL_PATH"


def load_model(model_path: str | None = None) -> Any:
    """
    Load and return your trained PyTorch UNet model.
    """
    default_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Model/checkouts/best_unet.pth")
    )
    path = (
        model_path
        if model_path and os.path.exists(model_path)
        else os.getenv(MODEL_ENV) or default_path
    )

    if not os.path.exists(path):
        print(f"Model file not found at {path}, returning None")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We load the weights
    state_dict = torch.load(path, map_location=device, weights_only=True)
    in_channels = state_dict["inc.double_conv.0.weight"].shape[1]

    model = UNet(n_channels=in_channels, n_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_event_data(model, idx: int, hours: int = 48) -> dict:
    if model is None:
        raise ValueError("Model is None")

    device = next(model.parameters()).device
    nc_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../Model/dataset/final_feature_stack_DYNAMIC_interpolated.nc",
        )
    )
    if not os.path.exists(nc_path):
        # Fallback to empty if dataset missing
        return {
            "probGrid": np.zeros((320, 400)).tolist(),
            "groundTruth": [],
            "initialFire": [],
        }

    cache_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../Model/" + CACHE_PATH.replace(".pkl", "_fi.pkl"),
        )
    )
    ds_loaded, feature_vars, total_steps = _load_ds(nc_path, include_fire_input=True)
    stats = _compute_global_stats(ds_loaded, feature_vars, cache_path)

    # 1. Base Prediction
    X_data = ds_loaded[feature_vars].isel(valid_time=idx).to_array(dim="channel").values
    min_v = stats["min"][:, None, None]
    max_v = stats["max"][:, None, None]

    X_data = np.clip(X_data, min_v, max_v)
    denominator = max_v - min_v + np.float32(1e-6)
    X_norm = (X_data - min_v) / denominator

    X_tensor = torch.from_numpy(X_norm).float().unsqueeze(0)
    _, _, h, w = X_tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))

    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(X_tensor)
            probs = torch.sigmoid(logits)

    probs = probs.squeeze().cpu().numpy()
    if pad_h > 0 or pad_w > 0:
        probs = probs[:h, :w]

    # 2. Extract Ground Truth Sequence (Sparse)
    ground_truth = []
    end_idx = min(idx + hours, total_steps)
    fire_sequence = (
        ds_loaded["MODIS_FIRE_T1"].isel(valid_time=slice(idx, end_idx)).values
    )

    for t in range(fire_sequence.shape[0]):
        rows, cols = np.where(fire_sequence[t] > 0)
        frame_coords = [[int(r), int(c)] for r, c in zip(rows, cols)]
        ground_truth.append(frame_coords)

    # Convert initial fire
    init_fire = ds_loaded["MODIS_FIRE_T1"].isel(valid_time=idx).values
    init_rows, init_cols = np.where(init_fire > 0)
    initial_fire = [[int(r), int(c)] for r, c in zip(init_rows, init_cols)]

    return {
        "probGrid": probs.tolist(),
        "groundTruth": ground_truth,
        "initialFire": initial_fire,
    }


def run_inference(model) -> np.ndarray:
    """Legacy function to maintain compatibility for /fire-grid endpoint"""
    return np.array(get_event_data(model, idx=271, hours=1)["probGrid"])
