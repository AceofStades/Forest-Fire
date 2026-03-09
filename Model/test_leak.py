import src.dataset
import torch

# Load the UNet training dataset setup
train_loader, val_loader, feature_vars = src.dataset.load_split_data(batch_size=1, include_fire_input=True)

# Let's inspect a few batches from validation to see what the target looks like compared to input
for i, (inputs, targets) in enumerate(val_loader):
    if targets.sum() > 0:
        print(f"Batch {i}")
        
        # MODIS_FIRE_T1 is index 8 (from previous output)
        input_fire = inputs[0, 8]
        target_fire = targets[0, 0]
        
        print("Input Fire sum:", input_fire.sum().item())
        print("Target Fire sum:", target_fire.sum().item())
        print("Pixels that are EXACTLY the same (Input == Target):", (input_fire == target_fire).sum().item(), "out of", target_fire.numel())
        
        if i > 2:
            break

