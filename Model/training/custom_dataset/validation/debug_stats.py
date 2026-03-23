import pickle
with open("stats_cache_fi.pkl", "rb") as f:
    stats = pickle.load(f)
print("Mins:", stats["min"])
print("Maxs:", stats["max"])
