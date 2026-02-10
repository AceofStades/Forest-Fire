import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

INPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC.nc"
CACHE_PATH = "weights/stats_cache.pkl"


class FireDataset(Dataset):
    def __init__(self, data, targets, indices, seq_len, lead_time, augment=False):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.seq_len = seq_len
        self.lead_time = lead_time
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx]

        # 1. Time Stacking
        x_seq = self.data[t_idx - self.seq_len + 1 : t_idx + 1]
        x_flat = np.concatenate(list(x_seq), axis=0)

        # Target
        y_data = self.targets[t_idx + self.lead_time]

        x_tensor = torch.from_numpy(x_flat).float()
        y_tensor = torch.from_numpy(y_data).float().unsqueeze(0)

        # 2. Augmentation (Training Only)
        if self.augment:
            # Horizontal Flip
            if random.random() > 0.5:
                x_tensor = torch.flip(x_tensor, [-1])
                y_tensor = torch.flip(y_tensor, [-1])
            # Vertical Flip
            if random.random() > 0.5:
                x_tensor = torch.flip(x_tensor, [-2])
                y_tensor = torch.flip(y_tensor, [-2])
            # NO ROTATION (Prevents shape mismatch crash)

        # 3. Dynamic Padding
        _, h, w = x_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x_tensor = F.pad(x_tensor, (0, pad_w, 0, pad_h))
            y_tensor = F.pad(y_tensor, (0, pad_w, 0, pad_h))

        return x_tensor, y_tensor


def get_dataloaders(batch_size=32, seq_len=3, lead_time=1):
    print(f"Loading {INPUT_NC_PATH}...")
    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf") as ds:
        features = list(ds.data_vars)
        data_np = ds.to_array(dim="channel").values
        data_np = np.transpose(data_np, (1, 0, 2, 3)).astype(np.float32)
        target_np = ds["MODIS_FIRE_T1"].values.astype(np.float32)

    # Stats / Normalization
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = {
            "mean": np.nanmean(data_np, axis=(0, 2, 3), keepdims=True),
            "std": np.nanstd(data_np, axis=(0, 2, 3), keepdims=True),
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(stats, f)

    data_np = (data_np - stats["mean"]) / (stats["std"] + 1e-6)

    # --- RANDOM SPLIT (Fixes the Empty Validation Issue) ---
    print("Performing RANDOM Split (Shuffling time steps)...")

    total_samples = data_np.shape[0]
    indices = np.arange(seq_len - 1, total_samples - lead_time)

    # Deterministic Shuffle (So results are reproducible)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"Train Samples: {len(train_indices)} | Val Samples: {len(val_indices)}")

    # Both sets use the FULL MAP, but different time steps
    train_ds = FireDataset(
        data_np, target_np, train_indices, seq_len, lead_time, augment=True
    )
    val_ds = FireDataset(
        data_np, target_np, val_indices, seq_len, lead_time, augment=False
    )

    in_channels = len(features) * seq_len

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dl, val_dl, in_channels
