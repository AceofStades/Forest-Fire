import pandas as pd

df_orig = pd.read_csv("dataset/MODIS/modis_2016_India.csv")
df_final = pd.read_csv("dataset/MODIS/final-modis.csv")

print(f"Original shape: {df_orig.shape}")
print(f"Final shape: {df_final.shape}")

print("\n--- Original Date Range ---")
print(df_orig['acq_date'].min(), "to", df_orig['acq_date'].max())

print("\n--- Final Date Range ---")
print(df_final['acq_date'].min(), "to", df_final['acq_date'].max())

print("\n--- Original Bounding Box ---")
print(f"Lat: {df_orig['latitude'].min()} to {df_orig['latitude'].max()}")
print(f"Lon: {df_orig['longitude'].min()} to {df_orig['longitude'].max()}")

print("\n--- Final Bounding Box ---")
print(f"Lat: {df_final['latitude'].min()} to {df_final['latitude'].max()}")
print(f"Lon: {df_final['longitude'].min()} to {df_final['longitude'].max()}")

