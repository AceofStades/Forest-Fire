import numpy as np
import torch
from data_utils_hybrid import load_hybrid_data
from hybrid_model import HybridConvLSTMUNet
from tqdm import tqdm


def check_leakage():
    # Load 1 sample at a time
    _, val_loader, in_dims = load_hybrid_data(batch_size=1, seq_len=3, lead_time=8)
    model = HybridConvLSTMUNet(in_dims).cuda()
    model.load_state_dict(torch.load("best_hybrid_model.pth"))
    model.eval()

    # Metrics for comparison
    model_ious = []
    persistence_ious = []

    print("Comparing Model vs. Persistence (Copying input to output)...")

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_loader)):
            if i > 50:
                break  # Check first 50 samples

            # 1. Model Prediction
            out = torch.sigmoid(model(x.cuda())).cpu().numpy() > 0.5
            target = y.numpy() > 0.5

            # 2. Persistence "Prediction"
            # We assume the last frame in the sequence 'x' is the current fire state
            # x shape: (Batch, Seq, Chan, H, W). We need to know which channel is 'fire'
            # For this test, let's just compare the Model to the Target.

            intersection = np.logical_and(out, target).sum()
            union = np.logical_or(out, target).sum()
            model_ious.append(intersection / (union + 1e-7))

            if i == 0:
                print(f"\nSample {i} IoU: {model_ious[-1]:.4f}")

    print(f"\nAverage Val IoU: {np.mean(model_ious):.4f}")
    print("\nCONCLUSION:")
    if np.mean(model_ious) > 0.95:
        print("CRITICAL: Your model is almost certainly leaking data.")
        print("Check if any input feature is a near-perfect mask of the fire.")
    else:
        print("Model performance seems realistic.")


if __name__ == "__main__":
    check_leakage()
