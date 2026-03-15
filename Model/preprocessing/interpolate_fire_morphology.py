import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import os

INPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"
OUTPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"

def interpolate_fire(ds):
    """
    Replaces the 24-hour block-copy persistence with a sub-hourly morphological dilation.
    This creates 23 synthetic intermediate frames showing a fire gradually expanding over time.
    """
    print(f"Loading {INPUT_NC_PATH}...")
    ds = ds.load()
    fire = ds["MODIS_FIRE_T1"].values
    
    T, H, W = fire.shape
    new_fire = np.zeros_like(fire)
    
    print("Finding fire expansion events...")
    # Find indices where new fire is detected
    fire_sums = fire.sum(axis=(1, 2))
    
    # We look for steps where fire jumps up significantly (satellite pass)
    last_known_frame = np.zeros((H, W), dtype=np.float32)
    
    # We will iterate through time. If we see a jump, we look back and interpolate.
    for t in tqdm(range(T), desc="Processing Time Steps"):
        current_frame = fire[t]
        
        # If there's a big jump from yesterday, we interpolate the hours in between
        if current_frame.sum() > last_known_frame.sum() + 10:
            # Found a satellite update. Let's trace back to when we last saw a jump
            # and interpolate between last_known_frame and current_frame
            # For simplicity, let's assume the gap is up to 24 hours back.
            gap = 24
            start_t = max(0, t - gap)
            
            # Distance transform to find how far the new fire is from the old fire
            # We want to grow last_known_frame into current_frame over 'gap' steps
            mask_target = current_frame > 0
            mask_start = last_known_frame > 0
            
            # Areas that are new
            new_areas = mask_target & ~mask_start
            
            if new_areas.any():
                # Distance from the start fire
                # We invert the start mask so EDT calculates distance TO the start mask
                dist = distance_transform_edt(~mask_start)
                
                # Only care about distances inside the new target area
                max_dist = dist[new_areas].max() if new_areas.any() else 1
                
                for step in range(1, gap + 1):
                    # How much distance should have burned by this hour?
                    allowed_dist = (step / gap) * max_dist
                    
                    # New fire for this intermediate step
                    intermediate_mask = mask_start | (new_areas & (dist <= allowed_dist))
                    
                    # Assign to output array
                    idx = start_t + step
                    if idx <= t:
                        new_fire[idx] = np.maximum(new_fire[idx], intermediate_mask.astype(np.float32))
            
            last_known_frame = current_frame.copy()
            new_fire[t] = np.maximum(new_fire[t], current_frame)
            
        else:
            # Just carry forward the last known frame (it continues burning)
            new_fire[t] = np.maximum(new_fire[t], last_known_frame)
            # Update last known if current has more fire (edge cases)
            if current_frame.sum() > last_known_frame.sum():
                last_known_frame = current_frame.copy()
                
    print("Replacing dataset variable...")
    ds["MODIS_FIRE_T1"].values = new_fire
    
    print(f"Saving to {OUTPUT_NC_PATH}...")
    ds.to_netcdf(OUTPUT_NC_PATH, format="netcdf4", engine="h5netcdf")
    print("Done!")

if __name__ == "__main__":
    with xr.open_dataset(INPUT_NC_PATH, engine="h5netcdf") as ds:
        interpolate_fire(ds)
