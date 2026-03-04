import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_curve
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
from src.utils import calculate_accuracy, compute_metrics

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASTER_PATH = "dataset/final_feature_stack_MASTER.nc"

# --- 2. Monkey Patching ---
src.dataset.INPUT_NC_PATH = MASTER_PATH
legacy.data_utils_seq.INPUT_NC_PATH = MASTER_PATH
legacy.data_utils_hybrid.INPUT_NC_PATH = MASTER_PATH
legacy.data_utils_hybrid.CACHE_PATH = "stats_cache_hybrid.pkl"


# --- 3. Legacy Model Definition ---
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


def get_model_and_loader(model_path):
    model_name = Path(model_path).stem

    # Heuristic based on filename
    if "hybrid" in model_name:
        print("Detected HYBRID model architecture.")
        # Ensure correct cache usage
        if not os.path.exists("stats_cache_hybrid.pkl"):
            print("Computing hybrid stats (this may take a moment)...")

        _, val_loader, in_channels = legacy.data_utils_hybrid.load_hybrid_data(
            batch_size=8, seq_len=3, lead_time=8, split_mode="spatial"
        )
        model = HybridConvLSTMUNet(in_channels=in_channels)
        return model, val_loader, "hybrid"

    elif "convlstm" in model_name:
        print("Detected CONVLSTM model architecture.")
        _, val_loader, in_channels = legacy.data_utils_seq.load_seq_data(
            batch_size=8, seq_len=3
        )
        model = ConvLSTM(in_channels=in_channels, hidden_dims=[32, 32, 32])
        return model, val_loader, "convlstm"

    elif "fire_unet" in model_name:
        print("Detected LEGACY UNET model architecture.")
        _, val_loader, in_channels = src.dataset.load_split_data(batch_size=16)
        model = LegacyUNet(in_channels=in_channels, out_channels=1)
        return model, val_loader, "legacy_unet"

    else:
        print("Detected STANDARD UNET model architecture.")
        _, val_loader, in_channels = src.dataset.load_split_data(batch_size=16)
        model = UNet(n_channels=in_channels, n_classes=1)
        return model, val_loader, "unet"


def evaluate(model_path, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem

    print(f"Evaluating model: {model_path}")

    try:
        model, val_loader, arch_type = get_model_and_loader(model_path)
    except Exception as e:
        print(f"Failed to setup model/loader: {e}")
        return

    model = model.to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Attempting Strict=False loading as fallback...")
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE), strict=False
            )
            print("Weights loaded with strict=False.")
        except Exception as e2:
            print(f"Still failed: {e2}")
            return

    model.eval()

    total_iou, total_rec, total_acc = 0, 0, 0
    total_prec, total_f1 = 0, 0
    all_probs_list, all_targets_list = [], []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            if torch.isnan(inputs).any():
                continue

            logits = model(inputs)

            if arch_type == "legacy_unet":
                probs = logits
                # IMPORTANT: compute_metrics expects logits (it does sigmoid internally)
                # But LegacyUNet output IS ALREADY probabilities (Sigmoid applied).
                # So we must INVERSE sigmoid or modify compute_metrics behavior.
                # Easiest: Provide logit-like values to compute_metrics
                logits_for_metrics = torch.logit(probs.clamp(min=1e-6, max=1 - 1e-6))
            else:
                probs = torch.sigmoid(logits)
                logits_for_metrics = logits

            # Calculate Standard Metrics
            # Note: compute_metrics returns (iou, recall, max_conf) - wait,
            # checking src/utils.py again:
            # def compute_metrics(pred_logits, target, threshold=0.3):
            # returns iou, rec, max_conf
            # BUT in previous code I saw iou, rec, acc = compute_metrics(...)
            # src/utils.py says "Returns: IoU, Recall, Accuracy, Max_Confidence" in docstring but
            # code returns: return iou, rec, max_conf.
            # Wait, let's verify what `src/utils.py` actually returns.
            # It returns THREE values: iou, rec, max_conf.
            # My previous evaluate.py unpacks 3 values: iou, rec, acc.
            # This means `acc` variable was holding `max_conf`! That explains why it was 0.98+ often.
            # I should calculate Accuracy properly using calculate_accuracy.

            iou, rec, _ = compute_metrics(logits_for_metrics, targets, threshold=0.5)
            acc = calculate_accuracy(logits_for_metrics, targets, threshold=0.5)

            # Calculate Precision and F1 manually for the batch
            pred_bin = (probs > 0.5).float()
            tp = (pred_bin * targets).sum().item()
            fp = (pred_bin * (1 - targets)).sum().item()
            fn = ((1 - pred_bin) * targets).sum().item()

            precision = tp / (tp + fp + 1e-6)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

            total_iou += iou
            total_rec += rec
            total_acc += acc
            total_prec += precision
            total_f1 += f1

            # Subsample for plotting
            all_probs_list.append(probs.cpu().view(-1)[::100])
            all_targets_list.append(targets.cpu().view(-1)[::100])

    num_batches = len(val_loader)
    avg_iou = total_iou / num_batches
    avg_rec = total_rec / num_batches
    avg_acc = total_acc / num_batches
    avg_prec = total_prec / num_batches
    avg_f1 = total_f1 / num_batches

    print(f"\nFINAL RESULTS (Thresh=0.5):")
    print(f"Mean Accuracy:  {avg_acc:.4f}")
    print(f"Mean Precision: {avg_prec:.4f}")
    print(f"Mean Recall:    {avg_rec:.4f}")
    print(f"Mean F1 Score:  {avg_f1:.4f}")
    print(f"Mean IoU:       {avg_iou:.4f}")

    # Save results to text file
    with open(output_path / f"{model_name}_results.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Mean Accuracy:  {avg_acc:.4f}\n")
        f.write(f"Mean Precision: {avg_prec:.4f}\n")
        f.write(f"Mean Recall:    {avg_rec:.4f}\n")
        f.write(f"Mean F1 Score:  {avg_f1:.4f}\n")
        f.write(f"Mean IoU:       {avg_iou:.4f}\n")

    visualize_sample(
        model, val_loader, output_path / f"{model_name}_visual.png", arch_type
    )

    # Plots
    if all_probs_list:
        all_probs = torch.cat(all_probs_list).numpy()
        all_targets = torch.cat(all_targets_list).numpy()

        plt.figure(figsize=(12, 5))

        # ROC
        try:
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            roc_auc = auc(fpr, tpr)
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.title("ROC Curve")
            plt.legend()
        except Exception as e:
            print(f"Could not plot ROC: {e}")

        # PR
        try:
            precision, recall, _ = precision_recall_curve(all_targets, all_probs)
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color="blue", lw=2)
            plt.title("Precision-Recall Curve")
        except Exception as e:
            print(f"Could not plot PR: {e}")

        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_curves.png")
        print(f"Saved reports to {output_path}")
    else:
        print("No data collected for plots.")


def visualize_sample(model, loader, save_path, arch_type):
    print("Generating visual sample...")
    with torch.no_grad():
        for x, y in loader:
            if y.sum() > 50:  # Find a sample with fire
                x = x.to(DEVICE)

                logits = model(x)
                if arch_type == "legacy_unet":
                    pred = logits
                else:
                    pred = torch.sigmoid(logits)

                # Visualization logic
                if len(x.shape) == 5:  # Sequence data (B, T, C, H, W)
                    input_img = x[0, -1, 0].cpu().numpy()  # Channel 0
                else:
                    input_img = x[0, 0].cpu().numpy()  # Channel 0

                target_img = y[0, 0].cpu().numpy()
                pred_img = pred[0, 0].cpu().numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(input_img, cmap="inferno")
                plt.title("Input (Channel 0)")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(target_img, cmap="inferno")
                plt.title("Target Fire")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_img, cmap="inferno")
                plt.title("Prediction")
                plt.axis("off")

                plt.savefig(save_path)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to .pth model file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="images", help="Output directory for images"
    )
    args = parser.parse_args()

    evaluate(args.model, args.output)
