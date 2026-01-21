import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import load_split_data
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet

MODEL_PATH = "best_fire_unet.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(pred_prob, target, threshold=0.5):
    """Calculates IoU, Dice, Precision, Recall for a batch."""
    pred_bin = (pred_prob > threshold).float()

    # Flatten for calculation
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


def evaluate():
    print(f"Evaluating model on: {DEVICE}")

    _, val_loader, input_channels = load_split_data(batch_size=BATCH_SIZE)

    model = UNet(in_channels=input_channels, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Check the path.")
        return

    model.eval()

    total_iou = 0
    total_dice = 0
    total_prec = 0
    total_recall = 0
    num_batches = 0

    print("Running evaluation loop...")
    all_probs_list = []
    all_targets_list = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # 1. Standard Metrics (Threshold 0.5)
            iou, dice, prec, rec = calculate_metrics(probs, targets, threshold=0.5)
            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_recall += rec
            num_batches += 1

            # 2. Store Data (CPU side)
            # We must flatten here or later. Storing flat saves RAM immediately.
            # Downsample: Taking every 100th pixel is enough for a smooth curve
            # and prevents RAM explosion.
            all_probs_list.append(probs.cpu().view(-1)[::100])
            all_targets_list.append(targets.cpu().view(-1)[::100])

    # Combine into one massive 1D array
    all_probs = torch.cat(all_probs_list)
    all_targets = torch.cat(all_targets_list)

    print("\n--- Threshold Sweep ---")
    print(f"{'Threshold':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 46)

    # We need full data for the sweep, but since we downsampled for plots,
    # the metrics below are strictly for the downsampled population (approximate but accurate enough).
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
        pred_bin = (all_probs > thresh).float()

        tp = (pred_bin * all_targets).sum().item()
        fp = (pred_bin * (1 - all_targets)).sum().item()
        fn = ((1 - pred_bin) * all_targets).sum().item()

        epsilon = 1e-6
        iou = tp / (tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        print(f"{thresh:<10.2f} | {iou:<10.4f} | {precision:<10.4f} | {recall:<10.4f}")

    if num_batches == 0:
        print("Error: No batches processed.")
        return

    print("\n" + "=" * 30)
    print("FINAL EVALUATION RESULTS (Thresh=0.5)")
    print("=" * 30)
    print(f"IoU (Jaccard):    {total_iou / num_batches:.4f}")
    print(f"Dice (F1 Score):  {total_dice / num_batches:.4f}")
    print(f"Precision:        {total_prec / num_batches:.4f}")
    print(f"Recall:           {total_recall / num_batches:.4f}")
    print("=" * 30)

    # --- PLOTTING ---
    # Convert to Numpy for Scikit-Learn
    # .flatten() ensures 1D array, though .view(-1) above usually handles it.
    all_targets_np = all_targets.numpy().flatten()
    all_probs_np = all_probs.numpy().flatten()

    plt.figure(figsize=(12, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_targets_np, all_probs_np)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 1)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_targets_np, all_probs_np)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("\nSaved evaluation plots to 'evaluation_report.png'")


if __name__ == "__main__":
    evaluate()
