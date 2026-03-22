import src.dataset

train_loader, val_loader, feature_vars = src.dataset.load_split_data(batch_size=1, include_fire_input=True)

diff_count = 0
for inputs, targets in val_loader:
    input_fire = inputs[0, 8].round()
    target_fire = targets[0, 0]
    if (input_fire != target_fire).sum() > 0:
        diff_count += 1

print(f"Batches where Target is NOT EXACTLY equal to Input: {diff_count} out of {len(val_loader)}")
