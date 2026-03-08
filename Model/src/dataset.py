import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"
CACHE_PATH = "stats_cache.pkl"


class FireDataset(Dataset):
    """Single-frame dataset for UNet / AttentionUNet."""

    def __init__(self, loaded_ds, indices, stats, feature_vars):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx]

        X_data = (
            self.ds[self.feature_vars]
            .isel(valid_time=t_idx)
            .to_array(dim="channel")
            .values
        )
        Y_data = self.ds["MODIS_FIRE_T1"].isel(valid_time=t_idx + 1).values

        # Robust Normalization with Clipping
        min_v = self.stats["min"][:, None, None]
        max_v = self.stats["max"][:, None, None]

        X_data = np.clip(X_data, min_v, max_v)
        denominator = max_v - min_v + np.float32(1e-6)
        X_norm = (X_data - min_v) / denominator

        X_tensor = torch.from_numpy(X_norm).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


class FireSeqDataset(Dataset):
    """Sequence dataset for ConvLSTM / Hybrid models. Returns (T, C, H, W)."""

    def __init__(self, loaded_ds, indices, stats, feature_vars, seq_len=4):
        self.ds = loaded_ds
        self.indices = indices
        self.stats = stats
        self.feature_vars = feature_vars
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        target_t_idx = self.indices[idx]
        start_t = target_t_idx - self.seq_len + 1
        time_indices = [max(0, t) for t in range(start_t, target_t_idx + 1)]

        min_v = self.stats["min"][:, None, None]
        max_v = self.stats["max"][:, None, None]
        denom = max_v - min_v + np.float32(1e-6)

        frames = []
        for t in time_indices:
            data = (
                self.ds[self.feature_vars]
                .isel(valid_time=t)
                .to_array(dim="channel")
                .values
            )
            data = np.clip(data, min_v, max_v)
            frames.append((data - min_v) / denom)

        X_seq = np.stack(frames, axis=0)  # (T, C, H, W)
        Y_data = self.ds["MODIS_FIRE_T1"].isel(valid_time=target_t_idx + 1).values

        X_tensor = torch.from_numpy(X_seq).float()
        Y_tensor = torch.from_numpy(Y_data).float().unsqueeze(0)

        _, _, h, w = X_tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            X_tensor = F.pad(X_tensor, (0, pad_w, 0, pad_h))
            Y_tensor = F.pad(Y_tensor, (0, pad_w, 0, pad_h))

        return X_tensor, Y_tensor


# ---- Shared helpers ----

def _resolve_path(nc_path):
    """Try to find the dataset file."""
    if os.path.exists(nc_path):
        return nc_path
    alt = os.path.join("Model", nc_path)
    if os.path.exists(alt):
        return alt
    return nc_path  # let it fail with a clear path


def _compute_global_stats(ds_loaded, feature_vars, cache_path):
    if os.path.exists(cache_path):
        print(f"--- Loading cached stats from {cache_path} ---")
        try:
            with open(cache_path, "rb") as f:
                stats = pickle.load(f)
            if isinstance(stats, dict) and "min" in stats and "max" in stats:
                if len(stats["min"]) == len(feature_vars):
                    return stats
                print("Cache channel count mismatch. Recomputing...")
        except Exception as e:
            print(f"Warning: cache load failed ({e}). Recomputing...")

    print("--- Computing Robust Stats (2nd/98th Percentiles) ---")
    num_channels = len(feature_vars)
    mins = np.zeros(num_channels)
    maxs = np.zeros(num_channels)

    for i, var in enumerate(tqdm(feature_vars, desc="Analyzing Channels")):
        data = ds_loaded[var].values
        mins[i] = np.nanpercentile(data, 2)
        maxs[i] = np.nanpercentile(data, 98)

    stats = {"min": mins.astype(np.float32), "max": maxs.astype(np.float32)}
    with open(cache_path, "wb") as f:
        pickle.dump(stats, f)
    return stats


def _load_ds(nc_path=None, include_fire_input=False):
    """Load dataset into RAM. Returns (ds_loaded, feature_vars, total_steps).
    If include_fire_input=True, MODIS_FIRE_T1 at time T is included as a feature
    (autoregressive mode — helps the model learn fire dynamics)."""
    path = _resolve_path(nc_path or INPUT_NC_PATH)
    print(f"--- Loading dataset into RAM ({path}) ---")
    with xr.open_dataset(path, engine="h5netcdf", chunks=None) as ds:
        ds_loaded = ds.load()
        if include_fire_input:
            feature_vars = list(ds.data_vars)
        else:
            feature_vars = [v for v in ds.data_vars if v != "MODIS_FIRE_T1"]
        total_steps = ds.sizes["valid_time"]
    fire_frames = int((ds_loaded["MODIS_FIRE_T1"].values.sum(axis=(1, 2)) > 0).sum())
    print(f"--- Loaded: {total_steps} steps, {len(feature_vars)} features, "
          f"{fire_frames} fire frames ({100*fire_frames/total_steps:.1f}%) ---")
    if include_fire_input:
        print("--- MODIS_FIRE_T1 included as input feature (autoregressive mode) ---")
    return ds_loaded, feature_vars, total_steps


def _compute_sample_weights(ds_loaded, indices, fire_oversample_ratio=10.0):
    """Per-sample weights for WeightedRandomSampler.

    Fire-target frames (where MODIS_FIRE_T1 at t+1 has any fire) get
    ``fire_oversample_ratio`` weight; non-fire frames get 1.0.
    With ratio=10 and ~8.6 % fire frames this yields ~48 % fire in each epoch.
    """
    fire_data = ds_loaded["MODIS_FIRE_T1"].values
    fire_per_frame = fire_data.sum(axis=(1, 2))

    weights = []
    n_fire = 0
    for idx in indices:
        target_idx = idx + 1
        if target_idx < len(fire_per_frame) and fire_per_frame[target_idx] > 0:
            weights.append(fire_oversample_ratio)
            n_fire += 1
        else:
            weights.append(1.0)

    print(f"--- Weighted sampling: {n_fire}/{len(indices)} fire-target frames "
          f"({100 * n_fire / max(len(indices), 1):.1f}%), "
          f"oversample_ratio={fire_oversample_ratio} ---")
    return torch.DoubleTensor(weights)


# ---- Public loaders ----

def load_split_data(batch_size=32, nc_path=None, weighted_sampling=False,
                    fire_oversample_ratio=10.0, include_fire_input=False):
    """Single-frame loader for UNet / AttentionUNet.

    Args:
        weighted_sampling: oversample fire-containing target frames.
        fire_oversample_ratio: weight multiplier for fire frames in sampler.
        include_fire_input: include MODIS_FIRE_T1 at time T as an input channel.
    """
    ds_loaded, feature_vars, total_steps = _load_ds(nc_path, include_fire_input=include_fire_input)
    cache = CACHE_PATH if nc_path is None else CACHE_PATH.replace(".pkl", f"_{hash(nc_path) % 10000}.pkl")
    if include_fire_input:
        cache = cache.replace(".pkl", "_fi.pkl")
    stats = _compute_global_stats(ds_loaded, feature_vars, cache)

    indices = np.arange(total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireDataset(ds_loaded, train_idx, stats, feature_vars)
    val_ds = FireDataset(ds_loaded, val_idx, stats, feature_vars)

    nw = min(8, os.cpu_count() or 4)

    if weighted_sampling:
        train_weights = _compute_sample_weights(ds_loaded, train_idx, fire_oversample_ratio)
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=nw, pin_memory=True, prefetch_factor=4, persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True, prefetch_factor=4, persistent_workers=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, prefetch_factor=2, persistent_workers=True,
    )
    return train_loader, val_loader, len(feature_vars)


def load_seq_data(batch_size=4, seq_len=4, nc_path=None, weighted_sampling=False,
                  fire_oversample_ratio=10.0, include_fire_input=False):
    """Sequence loader for ConvLSTM / Hybrid. Returns (B, T, C, H, W) batches."""
    ds_loaded, feature_vars, total_steps = _load_ds(nc_path, include_fire_input=include_fire_input)
    cache = CACHE_PATH if nc_path is None else CACHE_PATH.replace(".pkl", f"_{hash(nc_path) % 10000}.pkl")
    if include_fire_input:
        cache = cache.replace(".pkl", "_fi.pkl")
    stats = _compute_global_stats(ds_loaded, feature_vars, cache)

    indices = np.arange(seq_len, total_steps - 1)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_ds = FireSeqDataset(ds_loaded, train_idx, stats, feature_vars, seq_len)
    val_ds = FireSeqDataset(ds_loaded, val_idx, stats, feature_vars, seq_len)

    nw = min(8, os.cpu_count() or 4)

    if weighted_sampling:
        train_weights = _compute_sample_weights(ds_loaded, train_idx, fire_oversample_ratio)
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=nw, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader, len(feature_vars)
