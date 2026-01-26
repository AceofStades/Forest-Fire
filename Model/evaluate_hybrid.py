import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils_hybrid import load_hybrid_data
from hybrid_model import HybridConvLSTMUNet
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "best_hybrid_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAD_TIME = 8  # Ensure this matches your training lead time
SEQ_LEN = 3  # Ensure this matches your training sequence length


def calculate_metrics(pred_prob, target, threshold=0.5):
    """Calculates IoU, Dice, Precision, and Recall for a batch."""
    pred_bin = (pred_prob > threshold).float()

    # Flatten for calculation
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    epsilon = 1e-7

    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    return iou, dice, precision, recall


def evaluate():
    print(f"Initializing Evaluation on: {DEVICE}")

    # Load loaders using the RAM-optimized utility
    _, val_loader, input_channels = load_hybrid_data(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, lead_time=LEAD_TIME
    )

    model = HybridConvLSTMUNet(in_channels=input_channels).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Hybrid Model loaded successfully.")

    total_iou, total_dice, total_prec, total_recall = 0, 0, 0, 0
    active_fire_recall = []
    num_batches = 0

    all_probs_list = []
    all_targets_list = []

    print(f"Running evaluation loop (Lead Time: {LEAD_TIME * 3}h)...")

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Testing"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # 1. Standard Metrics
            iou, dice, prec, rec = calculate_metrics(probs, targets, threshold=0.5)

            # 2. Specific Metric: Recall on frames that actually contain fire
            # This is important to see if the 0.99 IoU is real or background-bias
            has_fire = targets.sum(dim=(1, 2, 3)) > 0
            if has_fire.any():
                # Re-calculate recall only for those specific images in the batch
                fire_targets = targets[has_fire].view(-1)
                fire_preds = (probs[has_fire] > 0.5).float().view(-1)
                tp_fire = (fire_preds * fire_targets).sum().item()
                fn_fire = ((1 - fire_preds) * fire_targets).sum().item()
                active_fire_recall.append(tp_fire / (tp_fire + fn_fire + 1e-7))

            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_recall += rec
            num_batches += 1

            # 3. Store downsampled data for plots to avoid RAM overflow
            # We take every 50th pixel to maintain curve resolution while saving memory
            all_probs_list.append(probs.cpu().view(-1)[::50])
            all_targets_list.append(targets.cpu().view(-1)[::50])

    # Aggregate data
    all_probs = torch.cat(all_probs_list).numpy()
    all_targets = torch.cat(all_targets_list).numpy()

    print("\n" + "=" * 40)
    print("HYBRID MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Mean IoU:               {total_iou / num_batches:.4f}")
    print(f"Mean Dice (F1):         {total_dice / num_batches:.4f}")
    print(f"Global Precision:       {total_prec / num_batches:.4f}")
    print(f"Global Recall:          {total_recall / num_batches:.4f}")
    if active_fire_recall:
        print(f"Recall (Active Fire):   {np.mean(active_fire_recall):.4f}")
    print("-" * 40)

    # Threshold Sweep
    print("\n--- Sensitivity Analysis ---")
    print(f"{'Thresh':<8} | {'IoU':<8} | {'Prec':<8} | {'Recall':<8}")
    for t in [0.3, 0.5, 0.7, 0.9]:
        p_bin = (all_probs > t).astype(float)
        tp = (p_bin * all_targets).sum()
        fp = (p_bin * (1 - all_targets)).sum()
        fn = ((1 - p_bin) * all_targets).sum()
        tiou = tp / (tp + fp + fn + 1e-7)
        tprec = tp / (tp + fp + 1e-7)
        trec = tp / (tp + fn + 1e-7)
        print(f"{t:<8.1f} | {tiou:<8.4f} | {tprec:<8.4f} | {trec:<8.4f}")

    # Plotting
    plt.figure(figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Precision-Recall Curve
    prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(rec_curve, prec_curve)
    plt.subplot(1, 2, 2)
    plt.plot(
        rec_curve, prec_curve, color="teal", lw=2, label=f"PR (AUC = {pr_auc:.4f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("hybrid_evaluation_report.png")
    print("\nEvaluation complete. Report saved as 'hybrid_evaluation_report.png'")


if __name__ == "__main__":
    evaluate()
