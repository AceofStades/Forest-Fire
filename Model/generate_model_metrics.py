import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# --- 1. Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Switch CWD to .../Forest-Fire/Model

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Imports
import legacy.data_utils_hybrid
import legacy.data_utils_seq
import src.dataset
from legacy.convlstm_model import ConvLSTM
from legacy.hybrid_model import HybridConvLSTMUNet
from src.models import UNet

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: MASTER.nc has a static MODIS_FIRE_T1 — broken for training/eval. Use DYNAMIC_new.
DYNAMIC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"
OUTPUT_CSV = "weights/model_metrics.csv"
WEIGHTS_DIR = "weights"

# --- 2. Monkey Patching ---
src.dataset.INPUT_NC_PATH = DYNAMIC_PATH
legacy.data_utils_seq.INPUT_NC_PATH = DYNAMIC_PATH
legacy.data_utils_hybrid.INPUT_NC_PATH = DYNAMIC_PATH
legacy.data_utils_hybrid.CACHE_PATH = "stats_cache_hybrid.pkl"


# --- 3. Legacy Model Definition (for best_fire_unet.pth) ---
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class LegacyUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.dconv_down1 = double_conv(in_channels, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_bottom = double_conv(512, 1024)
        self.upconv_up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv_up1 = double_conv(1024, 512)
        self.upconv_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up2 = double_conv(512, 256)
        self.upconv_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up3 = double_conv(256, 128)
        self.upconv_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up4 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        u = self.dconv_bottom(x)
        u = self.upconv_up1(u)
        u = torch.cat([u, conv4], dim=1)
        u = self.dconv_up1(u)
        u = self.upconv_up2(u)
        u = torch.cat([u, conv3], dim=1)
        u = self.dconv_up2(u)
        u = self.upconv_up3(u)
        u = torch.cat([u, conv2], dim=1)
        u = self.dconv_up3(u)
        u = self.upconv_up4(u)
        u = torch.cat([u, conv1], dim=1)
        u = self.dconv_up4(u)
        out = self.conv_last(u)
        return self.sigmoid(out)


# --- 4. Evaluation Logic ---
def compute_metrics_batch(preds, targets, threshold=0.5):
    # Ensure preds are probabilities
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)

    if preds.max() > 1.0 or preds.min() < 0.0:
        probs = torch.sigmoid(preds)
    else:
        probs = preds

    preds_bin = (probs > threshold).float().cpu().numpy().flatten()
    targets_bin = targets.cpu().numpy().flatten()

    acc = accuracy_score(targets_bin, preds_bin)
    prec = precision_score(targets_bin, preds_bin, zero_division=0)
    rec = recall_score(targets_bin, preds_bin, zero_division=0)
    f1 = f1_score(targets_bin, preds_bin, zero_division=0)
    iou = jaccard_score(targets_bin, preds_bin, zero_division=0)

    return acc, prec, rec, f1, iou


def evaluate_model(model, loader, model_name):
    print(f"--- Evaluating {model_name} ---")
    model.eval()

    total_acc = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    total_iou = 0
    steps = 0

    pbar = tqdm(loader, desc=f"Eval {model_name}")

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if torch.isnan(x).any():
                continue

            try:
                outputs = model(x)
                acc, prec, rec, f1, iou = compute_metrics_batch(outputs, y)

                total_acc += acc
                total_prec += prec
                total_rec += rec
                total_f1 += f1
                total_iou += iou
                steps += 1
            except Exception as e:
                # Suppress batch errors to avoid spam, unless all fail
                continue

    if steps == 0:
        return 0, 0, 0, 0, 0

    return (
        total_acc / steps,
        total_prec / steps,
        total_rec / steps,
        total_f1 / steps,
        total_iou / steps,
    )


def main():
    results = []

    if not os.path.exists(WEIGHTS_DIR):
        print(f"Error: Weights directory '{WEIGHTS_DIR}' not found.")
        return

    # --- 1. Evaluate UNet Models ---
    print("\n>>> Loading Data for UNet (Standard) ...")
    try:
        _, val_loader_unet, in_channels_unet = src.dataset.load_split_data(
            batch_size=16
        )
        print(f"Detected Input Channels for UNet: {in_channels_unet}")

        # 1a. Standard UNet
        name = "best_unet.pth"
        path = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(path):
            print(f"Loading {name}...")
            model = UNet(n_channels=in_channels_unet, n_classes=1).to(DEVICE)
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                acc, prec, rec, f1, iou = evaluate_model(model, val_loader_unet, name)
                results.append(
                    {
                        "Model": name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "IoU": iou,
                    }
                )
            except Exception as e:
                print(f"Failed to load {name} with standard UNet: {str(e)}")
            del model

        # 1b. Legacy UNet (best_fire_unet.pth)
        name = "best_fire_unet.pth"
        path = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(path):
            print(f"Loading {name} (Legacy)...")
            model = LegacyUNet(in_channels=in_channels_unet, out_channels=1).to(DEVICE)
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                acc, prec, rec, f1, iou = evaluate_model(model, val_loader_unet, name)
                results.append(
                    {
                        "Model": name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "IoU": iou,
                    }
                )
            except Exception as e:
                print(f"Failed to load {name} with LegacyUNet: {str(e)}")
            del model

        torch.cuda.empty_cache()
        del val_loader_unet

    except Exception as e:
        print(f"Failed to setup UNet evaluation: {str(e)}")

    # --- 2. Evaluate ConvLSTM ---
    print("\n>>> Loading Data for ConvLSTM (Seq) ...")
    try:
        _, val_loader_seq, in_channels_seq = legacy.data_utils_seq.load_seq_data(
            batch_size=8, seq_len=3
        )
        name = "best_convlstm.pth"
        path = os.path.join(WEIGHTS_DIR, name)

        if os.path.exists(path):
            print(f"Loading {name}...")
            model = ConvLSTM(in_channels=in_channels_seq, hidden_dims=[32, 32, 32]).to(
                DEVICE
            )
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                acc, prec, rec, f1, iou = evaluate_model(model, val_loader_seq, name)
                results.append(
                    {
                        "Model": name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "IoU": iou,
                    }
                )
            except Exception as e:
                print(f"Failed to load {name}: {str(e)}")
            del model
            torch.cuda.empty_cache()

        del val_loader_seq
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Failed to setup ConvLSTM evaluation: {str(e)}")

    # --- 3. Evaluate Hybrid ---
    print("\n>>> Loading Data for Hybrid ...")
    try:
        if not os.path.exists("stats_cache_hybrid.pkl"):
            print("Computing new hybrid stats cache...")

        _, val_loader_hybrid, in_channels_hybrid = (
            legacy.data_utils_hybrid.load_hybrid_data(
                batch_size=8, seq_len=3, lead_time=8, split_mode="spatial"
            )
        )

        for name in ["best_hybrid_model.pth", "hybrid_fire_model.pth"]:
            path = os.path.join(WEIGHTS_DIR, name)
            if not os.path.exists(path):
                continue

            print(f"Loading {name}...")
            model = HybridConvLSTMUNet(in_channels=in_channels_hybrid).to(DEVICE)
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                acc, prec, rec, f1, iou = evaluate_model(model, val_loader_hybrid, name)
                results.append(
                    {
                        "Model": name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "IoU": iou,
                    }
                )
            except Exception as e:
                print(f"Failed to load {name}: {str(e)}")
            del model
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Failed to setup Hybrid evaluation: {str(e)}")

    # --- Save Results ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved metrics to {os.path.abspath(OUTPUT_CSV)}")
        print(df)
    else:
        print("\nNo models were evaluated successfully.")


if __name__ == "__main__":
    main()
