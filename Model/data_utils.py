import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import xarray as xr

INPUT_NC_PATH = "dataset/final_feature_stack.nc"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 4

class FireDataset(Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.X)

	def __getitem__(self.idx):
		return self.X[idx], self.Y[idx]


def load_split_data(input_path=INPUT_NC_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE, batch_size=BATCH_SIZE):
	print(f"Loading data from {input_path}...")
	ds = xr.open_dataset(input_path)

	feature_vars = [v for v in ds.data_vars if v not in ['MODIS_FIRE_T1']]

	# Temporal Shift: Features from Day t predict fire from Day t + 1
	modis_target = ds['MODIS_FIRE_T1'].shift(valid_time=-1)
	Y_labels = modis_target.isel(valid_time=slice(0, -1)).values
	X_features = ds[feature_vars].isel(valid_time=slice(0, -1))

	# Convert to Tensors and Normalize
	X_data = X_features.to_array(dim="channel").transpose("valid_time", "channel", "latitude", "longitude").values
	X_data_min = X_data.min(axis=(0, 2, 3), keepdims = True)
	X_data_max = X_data.max(axis = (0, 2, 3), keepdims = True)
	X_data_norm = (X_data - X_data_min) / (X_data_max - X_data_min + 1e-6)

	X_data_tensor = torch.tensor(X_data_norm, dtype = torch.float32)
	Y_labels_tensor = torch.tensor(Y_labels, dtype = torch.float32).unsqueeze(1)

	print("Total time steps avaiable for training: ", X_data_tensor.shape[0])
	print("Number of feature channels: ", X_data_tensor.shape[1])

	# Split
	X_train, X_val, Y_train, Y_val = train_test_split(X_data_tensor, Y_labels_tensor, test_size=test_size, random_state=random_state, shuffle=False)

	# DataLoaders
	train_loader = DataLoader(FireDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(FireDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, X_train.shape[1]
