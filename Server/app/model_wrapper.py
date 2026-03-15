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


def run_inference(model) -> np.ndarray:
    """
    Runs inference on a sample frame from the dataset to produce a prob grid.
    Returns a 320x400 (or similar) probability numpy array.
    """
    if model is None:
        raise ValueError("Model is None")

    device = next(model.parameters()).device

    # Load dataset
    # We will pick a specific index that has fire
    nc_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../Model/dataset/final_feature_stack_DYNAMIC_interpolated.nc",
        )
    )
    if not os.path.exists(nc_path):
        # Fallback to local sample if dataset is missing
        print(f"Dataset not found at {nc_path}, returning random data")
        return np.random.rand(320, 400).astype(np.float32)

    cache_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../Model/" + CACHE_PATH.replace(".pkl", "_fi.pkl"),
        )
    )
    ds_loaded, feature_vars, total_steps = _load_ds(nc_path, include_fire_input=True)
    stats = _compute_global_stats(ds_loaded, feature_vars, cache_path)

    # Pick an index with active fire (just an arbitrary one late in the dataset, or search for it)
    idx = 100  # arbitrary sample index

    X_data = ds_loaded[feature_vars].isel(valid_time=idx).to_array(dim="channel").values

    min_v = stats["min"][:, None, None]
    max_v = stats["max"][:, None, None]

    X_data = np.clip(X_data, min_v, max_v)
    denominator = max_v - min_v + np.float32(1e-6)
    X_norm = (X_data - min_v) / denominator

    X_tensor = torch.from_numpy(X_norm).float().unsqueeze(0)  # (1, C, H, W)

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

    # Remove padding if it was added
    if pad_h > 0 or pad_w > 0:
        probs = probs[:h, :w]

    return probs
