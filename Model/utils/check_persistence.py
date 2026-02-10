import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import get_dataloaders


def check_persistence():
    print("Checking how much the fire ACTUALLY moves in the dataset...")

    # 1. Load Validation Data
    _, val_loader, _ = get_dataloaders(batch_size=32, seq_len=3, lead_time=8)

    ious = []
    active_pixels = 0
    total_samples = 0

    print("\nComparing Input (Time T) vs Target (Time T+24h)...")

    for i, (x, y) in enumerate(tqdm(val_loader)):
        # x shape: [Batch, Channels, H, W] (Normalized)
        # y shape: [Batch, 1, H, W] (Binary 0/1)

        # 1. Recover Binary Mask from Normalized Input
        # The input 'x' is time-stacked. The last feature of the last frame is the current fire.
        # Index -1 gets the last channel (MODIS_FIRE at T=0).
        # Since x is Z-score normalized, Fire (1) will be > 0, No Fire (0) will be < 0.
        current_fire_normalized = x[:, -1, :, :].unsqueeze(1)
        current_fire_bin = (current_fire_normalized > 0.0).float()

        # 2. Calculate IoU per sample to handle batching correctly
        # We only care about samples that actually have fire in at least one frame
        # to avoid 0/0 division issues masking the real score.

        # Flatten for easy calculation
        pred = current_fire_bin.view(-1)
        target = y.view(-1)

        intersection = (pred * target).sum().item()
        union = pred.sum().item() + target.sum().item() - intersection

        if union > 0:
            iou = intersection / (union + 1e-6)
            ious.append(iou)
            active_pixels += target.sum().item()

        total_samples += x.size(0)

        # Check first 50 batches to get a statistically significant average
        if i >= 50:
            break

    if len(ious) == 0:
        print("\nWARNING: No fires found in the validation set batches checked.")
        return

    avg_iou = np.mean(ious)

    print("-" * 50)
    print(f"Validation Samples Checked: {total_samples}")
    print(f"Active Fire Pixels Found:   {int(active_pixels)}")
    print(f"Average Persistence IoU:    {avg_iou:.4f}")
    print("-" * 50)

    if avg_iou > 0.85:
        print("CONCLUSION: HIGH PERSISTENCE.")
        print("The fires in your validation set barely move in 24 hours.")
        print(f"The model getting {avg_iou:.2f}+ IoU is simply copying the input.")
        print("Action: This is acceptable for 'Detection', but to force 'Prediction',")
        print("you must use the delta target: (Target - Input).")
    elif avg_iou < 0.1:
        print("CONCLUSION: LOW PERSISTENCE.")
        print("The fires move significantly or disappear.")
        print("If your model gets 1.0 IoU here, it is suspicious.")
    else:
        print("CONCLUSION: BALANCED DYNAMICS.")
        print("Fires move/grow naturally.")


if __name__ == "__main__":
    check_persistence()
