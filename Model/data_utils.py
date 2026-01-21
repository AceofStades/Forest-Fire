import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_NC_PATH = "dataset/final_feature_stack_MASTER.nc"
CACHE_PATH = "stats_cache.pkl"


class FireDataset(Dataset):
    def __init__(self, loaded_ds, indices, stats, feature_vars):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx]

        # 1. Access Data from RAM
        X_data = (
            self.ds[self.feature_vars]
            .isel(valid_time=t_idx)
            .to_array(dim="channel")
            .values
        )
        Y_data = self.ds["MODIS_FIRE_T1"].isel(valid_time=t_idx + 1).values

        # 2. Vectorized Normalization (Prevent float64 promotion)
        min_v = self.stats["min"][:, None, None]
        max_v = self.stats["max"][:, None, None]

        # Cast epsilon to float32 to keep everything in 32-bit
        denominator = max_v - min_v + np.float32(1e-6)
        X_norm = (X_data - min_v) / denominator

        # 3. Convert to Tensor
        X_tensor = torch.from_numpy(X_norm).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        # 4. PAD to be divisible by 16 (Fixes the U-Net Crash)
        # We calculate how much padding we need for Height and Width
        _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            # Pad (Left, Right, Top, Bottom)
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


def compute_global_stats(ds_loaded, feature_vars):
    if os.path.exists(CACHE_PATH):
        print(f"--- Loading cached stats from {CACHE_PATH} ---")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("--- Computing stats on RAM resident data ---")
    num_channels = len(feature_vars)
    mins = np.zeros(num_channels)
    maxs = np.zeros(num_channels)

    for i, var in enumerate(tqdm(feature_vars, desc="Analyzing Channels")):
        data = ds_loaded[var].values
        mins[i] = np.nanmin(data)
        maxs[i] = np.nanmax(data)

    stats = {"min": mins.astype(np.float32), "max": maxs.astype(np.float32)}

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(stats, f)

    return stats


def load_split_data(batch_size=32):
    print(f"--- Loading entire dataset into RAM ({INPUT_NC_PATH}) ---")

    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf", chunks=None) as ds:
        ds_loaded = ds.load()  # Force load to RAM
        feature_vars = [v for v in ds.data_vars if v != "MODIS_FIRE_T1"]
        total_steps = ds.sizes["valid_time"]

    print("--- Dataset Loaded into RAM ---")

    stats = compute_global_stats(ds_loaded, feature_vars)

    indices = np.arange(total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireDataset(ds_loaded, train_idx, stats, feature_vars)
    val_ds = FireDataset(ds_loaded, val_idx, stats, feature_vars)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_loader, val_loader, len(feature_vars)
