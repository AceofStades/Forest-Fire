import src.dataset
import torch

train_loader, val_loader, feature_vars = src.dataset.load_split_data(batch_size=1, include_fire_input=True)

# Test correlation globally between target and inputs in val set
same_pixels = 0
total_pixels = 0
total_target_fires = 0
for inputs, targets in val_loader:
    input_fire = inputs[0, 8].round()
    target_fire = targets[0, 0]
    
    same_pixels += (input_fire == target_fire).sum().item()
    total_pixels += target_fire.numel()
    total_target_fires += target_fire.sum().item()

print(f"Total matching pixels (Input == Target): {same_pixels} / {total_pixels}")
print(f"Total Target Fires: {total_target_fires}")
