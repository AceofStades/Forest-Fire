import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the root Model directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.dataset import load_split_data
from src.models import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(SCRIPT_DIR, "checkouts/best_unet.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load dataloaders
    print("Loading validation data to compare with ground truth...")
    train_loader, val_loader, in_channels = load_split_data(
        batch_size=4, weighted_sampling=False, include_fire_input=True
    )

    # Load model
    print("Loading model weights...")
    model = UNet(n_channels=in_channels, n_classes=1)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Run a batch
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Ensure there is some fire in the targets
            if targets.sum() == 0:
                continue

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Take the first item in the batch
            input_img = (
                inputs[0, 0].cpu().numpy()
            )  # Just showing one channel as proxy for context
            target_mask = targets[0, 0].cpu().numpy()
            pred_mask = probs[0, 0].cpu().numpy()

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(input_img, cmap="gray")
            axes[0].set_title("Input (Channel 0)")
            axes[0].axis("off")

            axes[1].imshow(target_mask, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth (Actual Fire Spread)")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Model Prediction (Probability)")
            axes[2].axis("off")

            output_file = os.path.join(SCRIPT_DIR, "ground_truth_comparison.png")
            plt.savefig(output_file)
            print(f"Test case visual saved to {output_file}")
            break


if __name__ == "__main__":
    main()
