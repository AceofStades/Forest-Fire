import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

INPUT_NC_PATH = "dataset/final_feature_stack_MASTER.nc"
CACHE_PATH = "stats_cache.pkl"


class FireHybridDataset(Dataset):
    def __init__(self, data_array, target_array, indices, seq_len=3, lead_time=1):
        self.data = data_array
        self.targets = target_array
        self.indices = indices
        self.seq_len = seq_len
        self.lead_time = lead_time

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        target_t_idx = self.indices[idx]

        start_idx = target_t_idx - self.seq_len + 1
        end_idx = target_t_idx + 1

        # Slicing from RAM is nearly instant
        X_seq = self.data[start_idx:end_idx]
        y_idx = target_t_idx + self.lead_time
        Y_data = self.targets[y_idx]

        X_tensor = torch.from_numpy(X_seq).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        # Spatial Padding for UNet (16-pixel divisibility)
        _, _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


def load_hybrid_data(batch_size=32, seq_len=3, lead_time=1):
    print("Loading NetCDF into RAM...")

    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf") as ds:
        feature_vars = [v for v in ds.data_vars if v != "MODIS_FIRE_T1"]
        # Load and transpose to (Time, Channel, H, W)
        data_block = ds[feature_vars].to_array(dim="channel").values
        data_block = np.transpose(data_block, (1, 0, 2, 3)).astype(np.float32)
        target_block = ds["MODIS_FIRE_T1"].values.astype(np.float32)

    print(f"Data Loaded. Shape: {data_block.shape}")

    # Check cache for MEAN/STD specifically
    stats = None
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            temp_stats = pickle.load(f)
            if "mean" in temp_stats and "std" in temp_stats:
                stats = temp_stats
                print("Loaded Mean/Std from cache.")

    if stats is None:
        print("Computing new Mean/Std stats...")
        mean_v = np.nanmean(data_block, axis=(0, 2, 3), keepdims=True)
        std_v = np.nanstd(data_block, axis=(0, 2, 3), keepdims=True)
        stats = {"mean": mean_v, "std": std_v}
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(stats, f)

    print("Normalizing data in RAM...")
    # Standardize: (x - mu) / sigma
    data_block = (data_block - stats["mean"]) / (stats["std"] + 1e-6)

    total_steps = data_block.shape[0]
    # Indices must allow for history window and lead time lookahead
    all_indices = np.arange(seq_len - 1, total_steps - lead_time)

    split_point = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split_point]

    # Validation gap to prevent temporal leakage
    gap = seq_len + lead_time + 2
    val_indices = all_indices[split_point + gap :]

    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    train_ds = FireHybridDataset(
        data_block, target_block, train_indices, seq_len, lead_time
    )
    val_ds = FireHybridDataset(
        data_block, target_block, val_indices, seq_len, lead_time
    )

    # pin_memory=True and persistent_workers=True will maximize GPU throughput
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, len(feature_vars)
