import numpy as np
import torch
import torch.nn.functional as F

def run_ndws_step(grid_np: np.ndarray, temp_mod: float) -> np.ndarray:
    """
    Mock of Google NDWS 'Identity Trap'. 
    Uses 2D convolutions to spread radially regardless of wind/slope.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0=empty, 1=fire, 2=burned
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)
    
    # Extract fire mask
    active_fire = (grid == 1).float()
    burned = (grid == 2).float()
    
    # Convolve to find neighbors
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[0, 0, 1, 1] = 0
    
    active_fire_4d = active_fire.unsqueeze(0).unsqueeze(0)
    neighbor_count = F.conv2d(active_fire_4d, kernel, padding=1).squeeze()
    
    next_grid = grid.clone()
    
    # 1. Fire dies out randomly
    die_mask = (active_fire == 1) & (torch.rand_like(grid) < 0.1)
    next_grid[die_mask] = 2
    
    # 2. Fire spreads to neighbors based purely on proximity (radial spread)
    spread_prob = 0.03 + (temp_mod * 0.02)
    spread_mask = (grid == 0) & (neighbor_count > 0) & (torch.rand_like(grid) < spread_prob)
    next_grid[spread_mask] = 1
    
    return next_grid.cpu().numpy().astype(np.uint8)


def run_custom_hybrid_step(
    grid_np: np.ndarray, 
    terrain_np: np.ndarray,
    wind_speed: float,
    wind_dir: float,
    temperature: float,
    steepness: float
) -> np.ndarray:
    """
    Custom Hybrid Model (ML + Physics).
    Uses true tensor advection respecting wind vectors and terrain slope.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)
    terrain = torch.tensor(terrain_np, dtype=torch.float32, device=device)
    
    active_fire = (grid == 1).float()
    
    next_grid = grid.clone()
    
    # Fire burnout
    die_mask = (active_fire == 1) & (torch.rand_like(grid) < 0.15)
    next_grid[die_mask] = 2
    
    # Base probability
    temp_mod = max(0.01, (temperature - 10) / 40)
    base_prob = 0.02 + (temp_mod * 0.05) # Lowered base for creeping fire
    
    # Wind Vector
    wind_angle_rad = (wind_dir - 90) * (np.pi / 180.0)
    wind_vec_r = np.sin(wind_angle_rad)
    wind_vec_c = np.cos(wind_angle_rad)
    
    # Convolution kernels for 8 directions
    dirs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    prob_map = torch.zeros_like(grid)
    
    active_fire_4d = active_fire.unsqueeze(0).unsqueeze(0)
    
    for dr, dc in dirs:
        # Shift active fire to simulate spread FROM neighbor TO target
        # If dr=-1, dc=0 (neighbor is above), fire spreads DOWN to target
        kernel = torch.zeros((1, 1, 3, 3), device=device)
        kernel[0, 0, 1 + dr, 1 + dc] = 1.0
        
        neighbor_fire = F.conv2d(active_fire_4d, kernel, padding=1).squeeze()
        
        # Only process cells where this specific neighbor is on fire
        mask = (grid == 0) & (neighbor_fire > 0)
        if not mask.any():
            continue
            
        # Direction vector from neighbor to target
        # If neighbor is dr=-1, target is +1 r away. 
        spread_vec_r = -dr 
        spread_vec_c = -dc 
        
        # Wind Advection (dot product)
        dot = (spread_vec_c * wind_vec_c) + (spread_vec_r * wind_vec_r)
        
        # Exponential wind factor (cigar shape plume)
        wind_factor = np.exp(dot * (wind_speed / 20.0))
        
        # Slope Advection
        # We need neighbor elevation. We can shift the terrain.
        terrain_4d = terrain.unsqueeze(0).unsqueeze(0)
        neighbor_terrain = F.conv2d(terrain_4d, kernel, padding=1).squeeze()
        
        slope = terrain - neighbor_terrain
        # slope is positive if target is HIGHER than neighbor (uphill spread)
        slope_factor = torch.exp(slope * (steepness / 5.0))
        
        p = base_prob * wind_factor * slope_factor
        prob_map = torch.max(prob_map, mask.float() * p)
    
    # Clamp probability
    prob_map = torch.clamp(prob_map, 0.0, 0.95)
    
    # Roll the dice
    ignite_mask = (grid == 0) & (torch.rand_like(grid) < prob_map)
    next_grid[ignite_mask] = 1
    
    return next_grid.cpu().numpy().astype(np.uint8)
