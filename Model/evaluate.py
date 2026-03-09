import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

import src.dataset
from src.models import ConvLSTMFireNet, UNet
from src.utils import compute_best_threshold_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_type):
    model_path = f"checkouts/best_{model_type}.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print(f"Please run `uv run train.py --model {model_type}` first.")
        return

    print(f"\n--- Evaluating {model_type.upper()} ---")
    print(f"Loading weights from {model_path}...")

    if model_type == "unet":
        _, val_loader, in_channels = src.dataset.load_split_data(
            batch_size=16, include_fire_input=True
        )
        model = UNet(n_channels=in_channels, n_classes=1)
    elif model_type == "convlstm":
        _, val_loader, in_channels = src.dataset.load_seq_data(
            batch_size=8, seq_len=4, include_fire_input=True
        )
        model = ConvLSTMFireNet(
            in_channels=in_channels, n_classes=1, hidden_dims=[64, 64]
        )
    else:
        print("Unknown model type")
        return

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)
    model.eval()

    print("\nCalculating metrics across validation set...")

    # We will test thresholds from 0.05 to 0.5 to find the best operating point
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    metrics_per_thr = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating Batches"):
            inputs, targets = (
                inputs.to(DEVICE, non_blocking=True),
                targets.to(DEVICE, non_blocking=True),
            )
            if torch.isnan(inputs).any():
                continue

            with torch.amp.autocast("cuda"):
                probs = torch.sigmoid(model(inputs))

            # Move to CPU for metric calculation to avoid massive VRAM usage
            probs_np = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for t in thresholds:
                pred = (probs_np > t).astype(np.float32)
                metrics_per_thr[t]["tp"] += (pred * targets_np).sum()
                metrics_per_thr[t]["fp"] += (pred * (1 - targets_np)).sum()
                metrics_per_thr[t]["fn"] += ((1 - pred) * targets_np).sum()

    print("\n--- RESULTS ---")
    best_f1 = -1.0
    best_stats = {}

    for t in thresholds:
        tp = metrics_per_thr[t]["tp"]
        fp = metrics_per_thr[t]["fp"]
        fn = metrics_per_thr[t]["fn"]

        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_stats = {"thr": t, "f1": f1, "pr": pr, "rc": rc, "iou": iou}

    print(f"Optimal Threshold: {best_stats['thr']}")
    print(f"F1 Score:  {best_stats['f1']:.5f}")
    print(f"Precision: {best_stats['pr']:.5f}")
    print(f"Recall:    {best_stats['rc']:.5f}")
    print(f"IoU:       {best_stats['iou']:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["unet", "convlstm"]
    )
    args = parser.parse_args()
    evaluate_model(args.model)


if __name__ == "__main__":
    main()
