import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from src.dataset import _compute_global_stats, _load_ds, CACHE_PATH
from src.models import UNet, ConvLSTMFireNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_tensor(ds, feature_vars, stats, time_idx, seq_len=None):
    min_v = stats["min"][:, None, None]
    max_v = stats["max"][:, None, None]
    denom = max_v - min_v + np.float32(1e-6)

    if seq_len is None:
        # Single frame for UNet
        data = ds[feature_vars].isel(valid_time=time_idx).to_array(dim="channel").values
        data = np.clip(data, min_v, max_v)
        norm = (data - min_v) / denom
        tensor = torch.from_numpy(norm).float().unsqueeze(0)
    else:
        # Sequence for ConvLSTM
        start_t = time_idx - seq_len + 1
        time_indices = [max(0, t) for t in range(start_t, time_idx + 1)]
        frames = []
        for t in time_indices:
            data = ds[feature_vars].isel(valid_time=t).to_array(dim="channel").values
            data = np.clip(data, min_v, max_v)
            frames.append((data - min_v) / denom)
        seq = np.stack(frames, axis=0)
        tensor = torch.from_numpy(seq).float().unsqueeze(0)

    # Pad to mod 16
    if seq_len is None:
        _, _, h, w = tensor.shape
    else:
        _, _, _, h, w = tensor.shape

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
        
    return tensor.to(DEVICE), pad_h, pad_w


def visualize_spread(unet_path, convlstm_path, nc_path, time_idx=None):
    print("Loading data...")
    ds, feature_vars, total_steps = _load_ds(nc_path, include_fire_input=True)
    
    if time_idx is None:
        fire_sums = ds["MODIS_FIRE_T1"].values.sum(axis=(1,2))
        active_frames = np.where(fire_sums > 50)[0]
        if len(active_frames) == 0:
            print("No active fire frames found in dataset.")
            return
        time_idx = active_frames[len(active_frames) // 2]

    print(f"Testing on Time Index: {time_idx}")
    cache = CACHE_PATH.replace(".pkl", "_fi.pkl")
    stats = _compute_global_stats(ds, feature_vars, cache)

    print("Preparing Input Tensors...")
    unet_tensor, pad_h, pad_w = prepare_tensor(ds, feature_vars, stats, time_idx)
    convlstm_tensor, _, _ = prepare_tensor(ds, feature_vars, stats, time_idx, seq_len=4)

    print("Loading Models...")
    unet_model = UNet(n_channels=len(feature_vars), n_classes=1).to(DEVICE)
    unet_model.load_state_dict(torch.load(unet_path, map_location=DEVICE, weights_only=True))
    unet_model.eval()

    convlstm_model = ConvLSTMFireNet(in_channels=len(feature_vars), n_classes=1, hidden_dims=[64, 64]).to(DEVICE)
    if os.path.exists(convlstm_path):
        convlstm_model.load_state_dict(torch.load(convlstm_path, map_location=DEVICE, weights_only=True))
    else:
        print(f"Warning: {convlstm_path} not found. Proceeding with untrained ConvLSTM for demonstration.")
    convlstm_model.eval()

    print("Running Inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            unet_probs = torch.sigmoid(unet_model(unet_tensor)).squeeze().cpu().numpy()
            convlstm_probs = torch.sigmoid(convlstm_model(convlstm_tensor)).squeeze().cpu().numpy()
    
    # Remove padding for visualization
    if pad_h > 0 or pad_w > 0:
        unet_probs = unet_probs[:-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]
        convlstm_probs = convlstm_probs[:-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]

    current_fire = ds["MODIS_FIRE_T1"].isel(valid_time=time_idx).values
    target_fire = ds["MODIS_FIRE_T1"].isel(valid_time=time_idx + 1).values
    actual_spread = np.clip(target_fire - current_fire, 0, 1)

    # Crop
    fire_coords = np.argwhere(current_fire > 0)
    if len(fire_coords) > 0:
        y_min, x_min = fire_coords.min(axis=0)
        y_max, x_max = fire_coords.max(axis=0)
        y_min = max(0, y_min - 20); y_max = min(current_fire.shape[0], y_max + 20)
        x_min = max(0, x_min - 20); x_max = min(current_fire.shape[1], x_max + 20)
    else:
        y_min, y_max, x_min, x_max = 0, current_fire.shape[0], 0, current_fire.shape[1]

    print("Plotting...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # --- ROW 1: UNET ---
    axes[0, 0].imshow(current_fire[y_min:y_max, x_min:x_max], cmap='inferno')
    axes[0, 0].set_title('Current Fire (Time T)')
    
    u10 = ds["u10"].isel(valid_time=time_idx).values[y_min:y_max, x_min:x_max]
    v10 = ds["v10"].isel(valid_time=time_idx).values[y_min:y_max, x_min:x_max]
    y_grid, x_grid = np.mgrid[0:u10.shape[0], 0:u10.shape[1]]
    axes[0, 0].quiver(x_grid[::2, ::2], y_grid[::2, ::2], u10[::2, ::2], v10[::2, ::2], color='white', alpha=0.5)

    axes[0, 1].imshow(actual_spread[y_min:y_max, x_min:x_max], cmap='Reds')
    axes[0, 1].set_title('Ground Truth Spread (T+1)')

    im_unet = axes[0, 2].imshow(unet_probs[y_min:y_max, x_min:x_max], cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('UNet Predicted Probability')
    plt.colorbar(im_unet, ax=axes[0, 2], fraction=0.046, pad=0.04)

    unet_sim = np.clip(current_fire + unet_probs, 0, 1)
    axes[0, 3].imshow(unet_sim[y_min:y_max, x_min:x_max], cmap='inferno')
    axes[0, 3].set_title('UNet Simulated Fire (T+1)')

    # --- ROW 2: CONVLSTM ---
    axes[1, 0].imshow(current_fire[y_min:y_max, x_min:x_max], cmap='inferno')
    axes[1, 0].set_title('Current Fire (Time T)')
    axes[1, 0].quiver(x_grid[::2, ::2], y_grid[::2, ::2], u10[::2, ::2], v10[::2, ::2], color='white', alpha=0.5)

    axes[1, 1].imshow(actual_spread[y_min:y_max, x_min:x_max], cmap='Reds')
    axes[1, 1].set_title('Ground Truth Spread (T+1)')

    im_lstm = axes[1, 2].imshow(convlstm_probs[y_min:y_max, x_min:x_max], cmap='viridis', vmin=0, vmax=1)
    axes[1, 2].set_title('ConvLSTM Predicted Probability')
    plt.colorbar(im_lstm, ax=axes[1, 2], fraction=0.046, pad=0.04)

    lstm_sim = np.clip(current_fire + convlstm_probs, 0, 1)
    axes[1, 3].imshow(lstm_sim[y_min:y_max, x_min:x_max], cmap='inferno')
    axes[1, 3].set_title('ConvLSTM Simulated Fire (T+1)')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    out_path = "assets/model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet", type=str, default="checkouts/best_unet.pth")
    parser.add_argument("--convlstm", type=str, default="checkouts/best_convlstm.pth")
    parser.add_argument("--nc", type=str, default="dataset/final_feature_stack_DYNAMIC_interpolated.nc")
    args = parser.parse_args()
    
    visualize_spread(args.unet, args.convlstm, args.nc)
