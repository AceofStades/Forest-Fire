import os

import torch
from src.dataset import load_split_data
from src.models import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, in_channels = load_split_data(
    batch_size=16, include_fire_input=True
)
model = UNet(n_channels=in_channels, n_classes=1)
model.load_state_dict(
    torch.load("Model/checkouts/best_unet.pth", map_location=DEVICE, weights_only=True)
)
model = model.to(DEVICE)
model.eval()

max_p = 0.0
for inputs, targets in val_loader:
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(inputs))
    if probs.max().item() > max_p:
        max_p = probs.max().item()

print(f"Global Max Prob across all batches: {max_p:.6f}")
