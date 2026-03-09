import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

import src.dataset
from src.models import ConvLSTMFireNet, UNet
from src.utils import compute_best_threshold_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_type, model_path):
    print(f"\nEvaluating {model_type} from {model_path}...")

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

    val_f1_accum = 0.0
    val_batches = 0
    all_metrics = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = (
                inputs.to(DEVICE, non_blocking=True),
                targets.to(DEVICE, non_blocking=True),
            )
            if torch.isnan(inputs).any():
                continue

            with torch.amp.autocast("cuda"):
                logits = model(inputs)

            metrics = compute_best_threshold_metrics(logits, targets)
            all_metrics.append(metrics)
            val_f1_accum += metrics["f1"]
            val_batches += 1

    avg_f1 = val_f1_accum / max(1, val_batches)

    # Calculate average metrics across all batches for best threshold
    # Note: A proper global evaluation would accumulate TP/FP/FN across all batches first.
    # But for a quick summary, average is fine. Let's do a quick global evaluation.

    global_tp, global_fp, global_fn = 0, 0, 0
    best_thr = 0.5
    # Finding majority best threshold across batches
    if all_metrics:
        best_thr = max(
            set([m["threshold"] for m in all_metrics]),
            key=[m["threshold"] for m in all_metrics].count,
        )

    print(f"Global Eval at Threshold {best_thr}:")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = (
                inputs.to(DEVICE, non_blocking=True),
                targets.to(DEVICE, non_blocking=True),
            )
            with torch.amp.autocast("cuda"):
                probs = torch.sigmoid(model(inputs))
            pred = (probs > best_thr).float()
            global_tp += (pred * targets).sum().item()
            global_fp += (pred * (1 - targets)).sum().item()
            global_fn += ((1 - pred) * targets).sum().item()

    precision = global_tp / (global_tp + global_fp + 1e-8)
    recall = global_tp / (global_tp + global_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(
        f"--> {model_type} | F1={f1:.5f} | Prec={precision:.5f} | Rec={recall:.5f} | Thr={best_thr}"
    )
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["unet", "convlstm"]
    )
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    evaluate_model(args.model, args.path)


if __name__ == "__main__":
    main()
