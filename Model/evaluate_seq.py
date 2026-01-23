import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from convlstm_model import ConvLSTM
from data_utils_seq import load_seq_data
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

# --- CONFIG ---
MODEL_PATH = "best_convlstm.pth"
BATCH_SIZE = 8
SEQUENCE_LENGTH = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Paths (Distinct from U-Net files)
SAMPLE_OUTPUT_PATH = "simulation_input/fire_prediction_sample_seq.npy"
VISUALIZATION_PATH = "simulation_input/prediction_visual_seq.png"


def calculate_metrics(pred_prob, target, threshold=0.5):
    """Calculates IoU, Dice, Precision, Recall for a batch."""
    pred_bin = (pred_prob > threshold).float()

    # Flatten tensors
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
    print(f"Evaluating ConvLSTM on: {DEVICE}")
    os.makedirs("simulation_input", exist_ok=True)

    # 1. Load Sequence Data
    _, val_loader, input_channels = load_seq_data(BATCH_SIZE, SEQUENCE_LENGTH)

    # 2. Initialize Model
    model = ConvLSTM(
        in_channels=input_channels, out_channels=1, hidden_dims=[32, 32, 32]
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("ConvLSTM Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # Metrics Storage
    total_iou = 0
    total_dice = 0
    total_prec = 0
    total_recall = 0
    num_batches = 0

    all_probs_list = []
    all_targets_list = []

    print("Running evaluation loop...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Calculate Standard Metrics (Threshold 0.5)
            iou, dice, prec, rec = calculate_metrics(probs, targets, threshold=0.5)

            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_recall += rec
            num_batches += 1

            # Store for Threshold Sweep (CPU side)
            all_probs_list.append(probs.cpu().view(-1)[::100])
            all_targets_list.append(targets.cpu().view(-1)[::100])

    # Concatenate all batches
    all_probs = torch.cat(all_probs_list)
    all_targets = torch.cat(all_targets_list)

    # --- Threshold Sweep ---
    print("\n--- Threshold Sweep ---")
    print(f"{'Threshold':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 46)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_bin = (all_probs > thresh).float()

        tp = (pred_bin * all_targets).sum().item()
        fp = (pred_bin * (1 - all_targets)).sum().item()
        fn = ((1 - pred_bin) * all_targets).sum().item()

        epsilon = 1e-6
        iou = tp / (tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        print(f"{thresh:<10.2f} | {iou:<10.4f} | {precision:<10.4f} | {recall:<10.4f}")

    # --- Final Results ---
    print("\n" + "=" * 30)
    print("FINAL CONVLSTM RESULTS (Thresh=0.5)")
    print("=" * 30)
    print(f"IoU (Jaccard):    {total_iou / num_batches:.4f}")
    print(f"Dice (F1 Score):  {total_dice / num_batches:.4f}")
    print(f"Precision:        {total_prec / num_batches:.4f}")
    print(f"Recall:           {total_recall / num_batches:.4f}")
    print("=" * 30)

    # --- Plotting Reports ---
    all_targets_np = all_targets.numpy()
    all_probs_np = all_probs.numpy()

    plt.figure(figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(all_targets_np, all_probs_np)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 1)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (ConvLSTM)")
    plt.legend(loc="lower right")

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_targets_np, all_probs_np)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_report_seq.png")
    print("\nSaved evaluation plots to 'evaluation_report_seq.png'")

    # --- NEW: GENERATE VISUALIZATION & SAMPLE ---
    print("\nGenerating sample prediction visual...")

    # Grab a single batch from validation
    sample_inputs, sample_targets = next(iter(val_loader))
    sample_inputs = sample_inputs.to(DEVICE)

    with torch.no_grad():
        logits = model(sample_inputs)
        probs = torch.sigmoid(logits)

    # Extract first item in batch [0, 0, H, W] -> [H, W]
    prediction_np = probs[0, 0].cpu().numpy()
    target_np = sample_targets[0, 0].cpu().numpy()

    # Save .npy
    np.save(SAMPLE_OUTPUT_PATH, prediction_np)
    print(f"Saved simulation input to '{SAMPLE_OUTPUT_PATH}'")

    # Save Visual Comparison
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth (Sequence)")
    plt.imshow(target_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Prediction (ConvLSTM)")
    plt.imshow(prediction_np, cmap="hot", interpolation="nearest")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH)
    print(f"Saved visual comparison to '{VISUALIZATION_PATH}'")


if __name__ == "__main__":
    evaluate()
