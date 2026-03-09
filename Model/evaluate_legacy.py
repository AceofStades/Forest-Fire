import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "legacy"))

import src.dataset
from legacy.convlstm_model import ConvLSTM
from legacy.unet_model import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def evaluate_model(model, val_loader, model_name):
    print(f"\nEvaluating {model_name}...")
    model = model.to(DEVICE)
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
                if model_name == "Legacy_UNet":
                    probs = model(inputs)
                else:
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

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"f1": f1, "precision": pr, "recall": rc, "threshold": t}

    if best_metrics is None:
        best_metrics = {"f1": 0, "precision": 0, "recall": 0, "threshold": 0.5}

    print(
        f"--> {model_name} | F1={best_metrics['f1']:.5f} | Prec={best_metrics['precision']:.5f} | Rec={best_metrics['recall']:.5f} | Thr={best_metrics['threshold']}"
    )


def main():
    print("Testing Legacy UNet...")
    _, val_loader, in_channels = src.dataset.load_split_data(batch_size=32)
    unet = UNet(in_channels=in_channels, out_channels=1)
    unet.load_state_dict(torch.load("weights/best_fire_unet.pth", map_location=DEVICE))
    evaluate_model(unet, val_loader, "Legacy_UNet")

    print("\nTesting Legacy ConvLSTM...")
    _, val_loader, in_channels = src.dataset.load_seq_data(batch_size=16, seq_len=3)
    convlstm = ConvLSTM(
        in_channels=in_channels, out_channels=1, hidden_dims=[32, 32, 32]
    )
    convlstm.load_state_dict(
        torch.load("weights/best_convlstm.pth", map_location=DEVICE)
    )
    evaluate_model(convlstm, val_loader, "Legacy_ConvLSTM")


if __name__ == "__main__":
    main()
