import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import load_split_data
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet

# Default Configuration
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(pred_prob, target, threshold=0.5):
    """Calculates IoU, Dice, Precision, Recall for a batch."""
    pred_bin = (pred_prob > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    epsilon = 1e-6
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    return iou, dice, precision, recall


def evaluate(model_path, output_dir):
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate a prefix based on model name for files
    model_name = Path(model_path).stem

    print(f"Evaluating model: {model_path} on {DEVICE}")

    _, val_loader, input_channels = load_split_data(batch_size=BATCH_SIZE)

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    total_iou, total_dice, total_prec, total_recall = 0, 0, 0, 0
    num_batches = 0
    all_probs_list, all_targets_list = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            iou, dice, prec, rec = calculate_metrics(probs, targets, threshold=0.5)
            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_recall += rec
            num_batches += 1

            # Downsample for RAM efficiency in plotting
            all_probs_list.append(probs.cpu().view(-1)[::100])
            all_targets_list.append(targets.cpu().view(-1)[::100])

    all_probs = torch.cat(all_probs_list)
    all_targets = torch.cat(all_targets_list)

    # --- Metrics Table Construction ---
    report_lines = []
    report_lines.append(f"Model: {model_path}")
    report_lines.append("-" * 46)
    report_lines.append(
        f"{'Threshold':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}"
    )
    report_lines.append("-" * 46)

    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
        pred_bin = (all_probs > thresh).float()
        tp = (pred_bin * all_targets).sum().item()
        fp = (pred_bin * (1 - all_targets)).sum().item()
        fn = ((1 - pred_bin) * all_targets).sum().item()

        epsilon = 1e-6
        iou = tp / (tp + fp + fn + epsilon)
        prec = tp / (tp + fp + epsilon)
        rec = tp / (tp + fn + epsilon)
        report_lines.append(
            f"{thresh:<10.2f} | {iou:<10.4f} | {prec:<10.4f} | {rec:<10.4f}"
        )

    # Final Stats
    summary = (
        f"\nFINAL EVALUATION (Thresh=0.5)\n"
        f"IoU: {total_iou / num_batches:.4f}\n"
        f"Dice: {total_dice / num_batches:.4f}\n"
        f"Precision: {total_prec / num_batches:.4f}\n"
        f"Recall: {total_recall / num_batches:.4f}"
    )
    print(summary)

    # Save Text Report
    with open(output_path / f"{model_name}_report.txt", "w") as f:
        f.write("\n".join(report_lines) + summary)

    # --- PLOTTING ---
    all_targets_np = all_targets.numpy().flatten()
    all_probs_np = all_probs.numpy().flatten()

    plt.figure(figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(all_targets_np, all_probs_np)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC: {model_name}")
    plt.legend()

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_targets_np, all_probs_np)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(True)

    plot_file = output_path / f"{model_name}_plots.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"\nSaved report and plots to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Evaluation Script")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to the .pth model file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="reports", help="Directory to save outputs"
    )

    args = parser.parse_args()
    evaluate(args.model, args.output)
