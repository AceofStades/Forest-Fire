import os

import numpy as np
import torch
from src.dataset import load_split_data
from src.models import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")
train_loader, val_loader, in_channels = load_split_data(
    batch_size=4, include_fire_input=True
)

model = UNet(n_channels=in_channels, n_classes=1)
model_path = "checkouts/best_unet.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
model = model.to(DEVICE)
model.eval()

found_fire = False
for inputs, targets in val_loader:
    if targets.sum() > 0:
        found_fire = True
        break

if not found_fire:
    print("No fire found in validation set!")
else:
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)

    print(f"Logits shape: {logits.shape}")
    print(
        f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}"
    )
    print(
        f"Probs min: {probs.min().item():.4f}, max: {probs.max().item():.4f}, mean: {probs.mean().item():.4f}"
    )

    # Target stats
    print(
        f"Targets sum: {targets.sum().item()}, min: {targets.min().item()}, max: {targets.max().item()}"
    )

    for b in range(inputs.shape[0]):
        target_sum = targets[b].sum().item()
        pred_sum_05 = (probs[b] > 0.5).sum().item()
        pred_sum_01 = (probs[b] > 0.1).sum().item()
        pred_sum_001 = (probs[b] > 0.01).sum().item()

        # Max prob in fire regions vs non-fire regions
        has_fire = targets[b] > 0
        if has_fire.sum() > 0:
            max_prob_fire = probs[b][has_fire].max().item()
            mean_prob_fire = probs[b][has_fire].mean().item()
        else:
            max_prob_fire = 0
            mean_prob_fire = 0

        max_prob_non_fire = (
            probs[b][~has_fire].max().item() if (~has_fire).sum() > 0 else 0
        )

        print(
            f"Batch {b}: Target fire pixels = {target_sum}, "
            f"Preds > 0.5 = {pred_sum_05}, > 0.1 = {pred_sum_01}, > 0.01 = {pred_sum_001}"
        )
        print(
            f"  Max prob fire region: {max_prob_fire:.4f}, Mean prob fire region: {mean_prob_fire:.4f}"
        )
        print(f"  Max prob NON-fire region: {max_prob_non_fire:.4f}")
