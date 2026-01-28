import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils_hybrid import load_hybrid_data


def check_dataset_reality():
    # Load validation data
    print("Loading Validation Data...")
    _, val_loader, _ = load_hybrid_data(batch_size=1, seq_len=3, lead_time=8)

    persistence_scores = []

    print("Checking Dataset 'Boringness' (Persistence)...")
    # We want to check: IoU(Fire at T, Fire at T+LeadTime)
    # Since we stripped Fire from inputs, we can't easily compare X to Y directly
    # UNLESS we assume the model learned to map a feature (like cvl) to the fire.

    # Instead, let's visualize the correlation between Features and Target

    # Get a sample with fire
    fire_sample_found = False

    for i, (x, y) in enumerate(val_loader):
        # y shape: (1, 1, H, W)
        if y.sum() > 100:  # Only look at frames with decent fire
            fire_sample_found = True

            # Extract features (Batch 0, Seq -1 (latest), Channel ?, H, W)
            # We need to find 'cvl' or 'swvl1'.
            # Based on your logs: ['d2m', 't2m', 'swvl1', 'e', 'u10', 'v10', 'tp', 'cvl', 'DEM', 'LULC', 'GHS_BUILT']
            # Indices: swvl1=2, cvl=7

            x_np = x.numpy()[0, -1]  # Latest frame in sequence
            y_np = y.numpy()[0, 0]  # Target (Future Fire)

            cvl_layer = x_np[7]  # Vegetation
            swvl_layer = x_np[2]  # Soil Moisture

            # Normalize for visualization
            cvl_norm = (cvl_layer - cvl_layer.min()) / (
                cvl_layer.max() - cvl_layer.min()
            )

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(y_np, cmap="inferno")
            plt.title("Target Fire (T + 24h)")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cvl_norm, cmap="Greens")
            plt.title("Input Vegetation 'cvl' (T)")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(swvl_layer, cmap="Blues")
            plt.title("Input Soil Moisture 'swvl1' (T)")
            plt.axis("off")

            plt.suptitle(f"Sample {i}: Checking for Feature Leakage")
            plt.tight_layout()
            plt.show()

            # Calculate correlation just to see
            flat_y = y_np.flatten()
            flat_cvl = cvl_layer.flatten()
            corr = np.corrcoef(flat_y, flat_cvl)[0, 1]
            print(f"Sample {i} - Correlation between Fire and Vegetation: {corr:.4f}")

            if i > 2:
                break  # Check 3 samples

    if not fire_sample_found:
        print("Could not find a sample with significant fire to visualize.")


if __name__ == "__main__":
    check_dataset_reality()
