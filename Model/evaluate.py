import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from src.dataset import get_dataloaders
from src.models import UNet
from src.utils import compute_metrics
from tqdm import tqdm

SEQ_LEN = 3
LEAD_TIME = 8
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem

    print(f"Evaluating model: {model_path}")

    _, val_loader, in_channels = get_dataloaders(BATCH_SIZE, SEQ_LEN, LEAD_TIME)

    model = UNet(n_channels=in_channels, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    total_iou, total_rec, total_acc = 0, 0, 0
    all_probs_list, all_targets_list = [], []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Updated to unpack 3 values
            iou, rec, acc = compute_metrics(logits, targets, threshold=0.5)
            total_iou += iou
            total_rec += rec
            total_acc += acc

            all_probs_list.append(probs.cpu().view(-1)[::100])
            all_targets_list.append(targets.cpu().view(-1)[::100])

    avg_iou = total_iou / len(val_loader)
    avg_rec = total_rec / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    print(f"\nFINAL RESULTS (Thresh=0.5):")
    print(f"Mean IoU:      {avg_iou:.4f}")
    print(f"Mean Recall:   {avg_rec:.4f}")
    print(f"Mean Accuracy: {avg_acc:.4f}")

    visualize_sample(model, val_loader, output_path / f"{model_name}_visual.png")

    # Plots
    all_probs = torch.cat(all_probs_list).numpy()
    all_targets = torch.cat(all_targets_list).numpy()

    plt.figure(figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.title("ROC Curve")
    plt.legend()

    # PR
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_curves.png")
    print(f"Saved reports to {output_path}")


def visualize_sample(model, loader, save_path):
    print("Generating visual sample...")
    with torch.no_grad():
        for x, y in loader:
            if y.sum() > 50:
                x = x.to(DEVICE)
                pred = torch.sigmoid(model(x))

                input_img = x[0, -1].cpu().numpy()
                target_img = y[0, 0].cpu().numpy()
                pred_img = pred[0, 0].cpu().numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(input_img, cmap="inferno")
                plt.title("Input (Current Fire)")
                plt.subplot(1, 3, 2)
                plt.imshow(target_img, cmap="inferno")
                plt.title("Target (T+24h)")
                plt.subplot(1, 3, 3)
                plt.imshow(pred_img, cmap="inferno")
                plt.title("Prediction")
                plt.savefig(save_path)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="weights/best_unet.pth")
    parser.add_argument("-o", "--output", type=str, default="simulation_input")
    args = parser.parse_args()

    evaluate(args.model, args.output)
