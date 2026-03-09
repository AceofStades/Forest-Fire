import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- 1. Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Switch CWD to .../Forest-Fire/Model

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Strictly use new modules
import src.dataset
from src.models import AttentionUNet, ConvLSTMFireNet, HybridFireNet, UNet

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DYNAMIC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"

# Patch dataset path
src.dataset.INPUT_NC_PATH = DYNAMIC_PATH


def get_model_and_loader(model_path):
    model_name = Path(model_path).stem

    if "hybrid" in model_name:
        seq_len = 4 if "_s4" in model_name else 6
        include_fire_input = "_fi" in model_name
        _, val_loader, in_channels = src.dataset.load_seq_data(
            batch_size=16, seq_len=seq_len, include_fire_input=include_fire_input
        )
        model = HybridFireNet(in_channels=in_channels, n_classes=1, base_filters=32)
        return model, val_loader

    elif "lstm" in model_name or "convlstm" in model_name:
        seq_len = 4 if "_s4" in model_name else 6
        include_fire_input = "_fi" in model_name
        _, val_loader, in_channels = src.dataset.load_seq_data(
            batch_size=16, seq_len=seq_len, include_fire_input=include_fire_input
        )
        model = ConvLSTMFireNet(
            in_channels=in_channels, n_classes=1, hidden_dims=[64, 64]
        )
        return model, val_loader

    elif "attn" in model_name:
        include_fire_input = "_fi" in model_name
        _, val_loader, in_channels = src.dataset.load_split_data(
            batch_size=32, include_fire_input=include_fire_input
        )
        model = AttentionUNet(n_channels=in_channels, n_classes=1, base_filters=32)
        return model, val_loader

    else:
        _, val_loader, in_channels = src.dataset.load_split_data(batch_size=32)
        model = UNet(n_channels=in_channels, n_classes=1)
        return model, val_loader


EVAL_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def evaluate(model_path):
    model_name = Path(model_path).stem
    print(f"\nEvaluating {model_name}...")

    try:
        model, val_loader = get_model_and_loader(model_path)
    except Exception as e:
        print(f"Failed to setup model/loader: {e}")
        return None

    model = model.to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE), strict=False
            )
        except Exception as e2:
            print(f"Failed to load weights: {e2}")
            return None

    model.eval()

    thr_acc = {t: [0.0, 0.0, 0.0] for t in EVAL_THRESHOLDS}  # tp, fp, fn
    global_max_prob = 0.0

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
                probs = torch.sigmoid(logits)

            batch_max = probs.max().item()
            if batch_max > global_max_prob:
                global_max_prob = batch_max

            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)

            for t in EVAL_THRESHOLDS:
                pred = (probs_flat > t).float()
                tp = (pred * targets_flat).sum().item()
                fp = (pred * (1 - targets_flat)).sum().item()
                fn = ((1 - pred) * targets_flat).sum().item()

                thr_acc[t][0] += tp
                thr_acc[t][1] += fp
                thr_acc[t][2] += fn

    best_f1, best_metrics = -1.0, None
    for t in EVAL_THRESHOLDS:
        tp, fp, fn = thr_acc[t]
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "f1": f1,
                "iou": iou,
                "precision": pr,
                "recall": rc,
                "threshold": t,
                "max_prob": global_max_prob,
            }

    if best_metrics is None:
        best_metrics = {
            "f1": 0,
            "iou": 0,
            "precision": 0,
            "recall": 0,
            "threshold": 0.5,
            "max_prob": global_max_prob,
        }

    print(
        f"--> {model_name} | F1={best_metrics['f1']:.5f} | Prec={best_metrics['precision']:.5f} | Rec={best_metrics['recall']:.5f} | Thr={best_metrics['threshold']}"
    )
    best_metrics["name"] = model_name
    return best_metrics


def main():
    import glob

    checkouts_dir = Path("checkouts")
    pth_files = glob.glob(str(checkouts_dir / "*.pth"))

    if not pth_files:
        print("No .pth files found in checkouts/")
        return

    print(f"Found {len(pth_files)} model files. Evaluating all...")

    results = []
    for pth in sorted(pth_files):
        res = evaluate(pth)
        if res:
            results.append(res)

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda x: x["f1"], reverse=True)

    csv_path = checkouts_dir / "sweep_summary.csv"
    print(f"\nWriting new summary to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "name",
                "f1",
                "iou",
                "precision",
                "recall",
                "threshold",
                "max_prob",
            ]
        )
        for idx, r in enumerate(results, 1):
            writer.writerow(
                [
                    idx,
                    r["name"],
                    f"{r['f1']:.6f}",
                    f"{r['iou']:.6f}",
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    r["threshold"],
                    f"{r['max_prob']:.6f}",
                ]
            )

    print("Done! Top 3 models:")
    for r in results[:3]:
        print(f" - {r['name']} | F1: {r['f1']:.5f}")


if __name__ == "__main__":
    main()
