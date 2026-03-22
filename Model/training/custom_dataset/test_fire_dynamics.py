import src.dataset
import torch

ds, feature_vars, total_steps = src.dataset._load_ds(include_fire_input=True)

print("Vars order:", feature_vars)
fire_idx = feature_vars.index("MODIS_FIRE_T1")
print(f"Fire index is {fire_idx}")

# Let's get the exact correlation between T and T+1
total_fire_t = 0
total_fire_t1 = 0
stayed_fire = 0

fire = ds["MODIS_FIRE_T1"].values

for t in range(fire.shape[0]-1):
    f_t = fire[t]
    f_t1 = fire[t+1]
    
    total_fire_t += f_t.sum()
    total_fire_t1 += f_t1.sum()
    stayed_fire += (f_t * f_t1).sum()

print(f"Total fire pixels at T: {total_fire_t}")
print(f"Total fire pixels at T+1: {total_fire_t1}")
print(f"Fire that stayed fire: {stayed_fire}")

if total_fire_t > 0:
    print(f"Percentage of fire that persisted to the next frame: {stayed_fire / total_fire_t * 100:.2f}%")

