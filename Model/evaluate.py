import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from data_utils import load_split_data
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet

MODEL_PATH = "best_fire_unet.pth"
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

    all_targets = []
    all_probs = []

    print("Running evaluation loop...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            iou, dice, prec, rec = calculate_metrics(probs, targets)

            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_recall += rec
            num_batches += 1

            if len(all_targets) < 1000000:
                all_targets.extend(targets.cpu().numpy().flatten()[::100])
                all_probs.extend(probs.cpu().numpy().flatten()[::100])

    print("\n" + "=" * 30)
    print("FINAL EVALUATION RESULTS")
    print("=" * 30)
    print(f"IoU (Jaccard):    {total_iou / num_batches:.4f}")
    print(f"Dice (F1 Score):  {total_dice / num_batches:.4f}")
    print(f"Precision:        {total_prec / num_batches:.4f}")
    print(f"Recall:           {total_recall / num_batches:.4f}")
    print("=" * 30)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    plt.figure(figsize=(12, 5))

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
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

    precision, recall, _ = precision_recall_curve(all_targets, all_probs)

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
