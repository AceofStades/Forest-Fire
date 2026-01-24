# import numpy as np
# import pickle
# import os

# def load_prediction_grid(file_path: str):
#     """Loads the 320x400 UNet output grid."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Missing model output: {file_path}")
#     # fire_prediction_sample.npy contains per-pixel probabilities
#     return np.load(file_path)

# def load_normalization_stats(file_path: str):
#     """Loads min/max stats for data consistency."""
#     with open(file_path, 'rb') as f:
#         # stats_cache.pkl stores the training distribution
#         stats = pickle.load(f)
#     return stats

import numpy as np

def generate_mock_fire_grid():
    # Create an empty grid (320 rows, 400 columns)
    # Default probability of fire = 0.1 (low risk)
    grid = np.full((320, 400), 0.1, dtype=np.float32)

    # Add a "Fire Front" (a rectangle of high risk)
    # This helps you see if D* Lite actually goes around it
    grid[100:220, 150:250] = 0.9  

    # Add some random "hotspots"
    random_spots = np.random.rand(320, 400)
    grid[random_spots > 0.98] = 0.8
    
    return grid

# Save it so your FastAPI can load it like a real file
# np.save('models/fire_prediction_sample.npy', generate_mock_fire_grid())