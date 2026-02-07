import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

INPUT_NC_PATH = "dataset/final_feature_stack_MASTER.nc"
CACHE_PATH = "stats_cache.pkl"
SEQUENCE_LENGTH = 3  # Look back 2 frames + current frame


class FireSeqDataset(Dataset):
    def __init__(self, loaded_ds, indices, stats, feature_vars, seq_len=3):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # The target time index
        target_t_idx = self.indices[idx]

        # Collect sequence of frames [t - seq_len + 1, ..., t]
        # Example: if seq_len=3 and target is 10, we want indices [8, 9, 10]
        start_t = target_t_idx - self.seq_len + 1

        # Safety: If index is negative, just repeat the first frame (padding in time)
        time_indices = [max(0, t) for t in range(start_t, target_t_idx + 1)]

        X_seq_list = []

        for t in time_indices:
            data = (
                self.ds[self.feature_vars]
                .isel(valid_time=t)
                .to_array(dim="channel")
                .values
            )

            # Robust Norm
            min_v = self.stats["min"][:, None, None]
            max_v = self.stats["max"][:, None, None]
            data = np.clip(data, min_v, max_v)
            denom = max_v - min_v + np.float32(1e-6)
            norm_data = (data - min_v) / denom

            X_seq_list.append(norm_data)

        # Stack into (Time, Channels, Height, Width)
        X_seq = np.stack(X_seq_list, axis=0)

        # Target is just the standard fire map at the LAST time step
        Y_data = self.ds["MODIS_FIRE_T1"].isel(valid_time=target_t_idx + 1).values

        X_tensor = torch.from_numpy(X_seq).float()  # (T, C, H, W)
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)  # (1, H, W)

        # Pad Spatial Dims to 320x400
        _, _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            # Pad last 2 dimensions (W, H)
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


def compute_global_stats(ds_loaded, feature_vars):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("--- Computing Stats (2nd/98th Percentiles) ---")
    mins = np.zeros(len(feature_vars))
    maxs = np.zeros(len(feature_vars))

    for i, var in enumerate(tqdm(feature_vars)):
        data = ds_loaded[var].values
        mins[i] = np.nanpercentile(data, 2)
        maxs[i] = np.nanpercentile(data, 98)

    stats = {"min": mins.astype(np.float32), "max": maxs.astype(np.float32)}
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(stats, f)
    return stats


def load_seq_data(batch_size=8, seq_len=3):
    print(f"--- Loading Dataset for Sequence Length {seq_len} ---")
    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf", chunks=None) as ds:
        ds_loaded = ds.load()
        feature_vars = [v for v in ds.data_vars if v != "MODIS_FIRE_T1"]
        total_steps = ds.sizes["valid_time"]

    stats = compute_global_stats(ds_loaded, feature_vars)

    # We start from index = seq_len so we have enough history
    indices = np.arange(seq_len, total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireSeqDataset(ds_loaded, train_idx, stats, feature_vars, seq_len)
    val_ds = FireSeqDataset(ds_loaded, val_idx, stats, feature_vars, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return train_loader, val_loader, len(feature_vars)
