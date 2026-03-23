import torch
import torch.nn as nn
from src.dataset import load_split_data
from src.models import UNet
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading data...")
    train_loader, val_loader, in_channels = load_split_data(batch_size=1, include_fire_input=True)

    model = UNet(n_channels=in_channels, n_classes=1)
    
    model_path = "checkouts/best_unet.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print("Loaded trained weights")
        
    model = model.to(DEVICE)
    model.eval()
    
    for inputs, targets in val_loader:
        if targets.sum() > 0:
            inputs = inputs.to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
                    probs = torch.sigmoid(logits)
            
            print("WITH AUTOCAST:")
            print(f"Logits min/max/mean: {logits.min().item():.4f}, {logits.max().item():.4f}, {logits.mean().item():.4f}")
            print(f"Probs min/max/mean: {probs.min().item():.4f}, {probs.max().item():.4f}, {probs.mean().item():.4f}")
            
            with torch.no_grad():
                logits = model(inputs)
                probs = torch.sigmoid(logits)
            
            print("WITHOUT AUTOCAST:")
            print(f"Logits min/max/mean: {logits.min().item():.4f}, {logits.max().item():.4f}, {logits.mean().item():.4f}")
            print(f"Probs min/max/mean: {probs.min().item():.4f}, {probs.max().item():.4f}, {probs.mean().item():.4f}")
            break

if __name__ == "__main__":
    main()
