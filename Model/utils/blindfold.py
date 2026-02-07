import numpy as np
import torch
from data_utils_hybrid import load_hybrid_data
from hybrid_model import HybridConvLSTMUNet
from tqdm import tqdm


def test_spatial_blindness():
    # 1. Load Normal Data
    print("Loading Validation Data...")
    _, val_loader, feature_count = load_hybrid_data(
        batch_size=32, seq_len=3, lead_time=8
    )

    model = HybridConvLSTMUNet(feature_count).cuda()
    model.load_state_dict(torch.load("best_hybrid_model.pth"))
    model.eval()

    # 2. Identify "Static" Channels to Zero Out
    # Based on your list: ['d2m', 't2m', 'swvl1', 'e', 'u10', 'v10', 'tp', 'cvl', 'DEM', 'LULC', 'GHS_BUILT']
    # Indices: DEM=8, LULC=9, GHS_BUILT=10
    static_indices = [8, 9, 10]

    print(f"Blindfolding model on channels: {static_indices} (DEM, LULC, GHS)...")

    original_iou = []
    blind_iou = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Running Blind Test"):
            x = x.cuda()
            y = y.cuda()

            # --- Pass 1: Original (With Map) ---
            pred_orig = torch.sigmoid(model(x))

            # --- Pass 2: Blindfolded (Zeroed Map) ---
            x_blind = x.clone()
            # Zero out the static features across all time steps and batch items
            for idx in static_indices:
                x_blind[:, :, idx, :, :] = 0.0

            pred_blind = torch.sigmoid(model(x_blind))

            # Calculate IoU for both
            def get_iou(pred, target):
                p = (pred > 0.5).float()
                t = (target > 0.5).float()
                inter = (p * t).sum()
                union = (p + t).sum() - inter
                return (inter / (union + 1e-6)).item()

            original_iou.append(get_iou(pred_orig, y))
            blind_iou.append(get_iou(pred_blind, y))

    print("\n" + "=" * 40)
    print("RESULTS: Is your model a Map Memorizer?")
    print("=" * 40)
    print(f"Original IoU (With Map):    {np.mean(original_iou):.4f}")
    print(f"Blind IoU (No Map):         {np.mean(blind_iou):.4f}")
    print("-" * 40)

    if np.mean(blind_iou) < 0.05:
        print("VERDICT: SPATIAL OVERFITTING DETECTED.")
        print("The model relies 100% on knowing WHERE it is on the map.")
        print("It has memorized the specific pixels that burned in the training set.")
    else:
        print("VERDICT: The model is actually using weather patterns!")


if __name__ == "__main__":
    test_spatial_blindness()
