# %%
import pandas as pd
from calendar import month

df = pd.read_csv("../dataset/MODIS/modis_2016_India.csv")
df.drop(columns=['instrument', 'version'], inplace=True)

north_lat = 31.5
south_lat = 28.7
west_lon = 77.5
east_lon = 81.1

mask = (df['latitude'] >= south_lat) & (df['latitude'] <= north_lat) & (df['longitude'] <= east_lon) & (df['longitude'] >= west_lon)
df_uttarakhand = df[mask].copy()

# %%
df_uttarakhand['acq_date'] = pd.to_datetime(df_uttarakhand['acq_date'])

# %%
april_mask = (df_uttarakhand['acq_date'].dt.month == 4)
may_mask = (df_uttarakhand['acq_date'].dt.month == 5)
df_april_may = df_uttarakhand[april_mask | may_mask].copy()

# %%
df_april_may['acq_timestamp'] = pd.to_datetime(
    df_april_may['acq_date'].astype(str) + ' ' +
    df_april_may['acq_time'].astype(str).str.zfill(4).str.slice(0, 2) + ':' +
    df_april_may['acq_time'].astype(str).str.zfill(4).str.slice(2, 4)
)

# %%
df_april_may_unique = df_april_may.drop_duplicates(subset=['acq_timestamp', 'latitude', 'longitude']).copy()

# %%
df_april_may_unique.to_csv('final-modis.csv', index=False)

print("Filtered MODIS data for April and May has been saved to 'final-modis.csv'.")
print("Duplicate date and time entries have been removed.")
print(f"Original Uttarakhand-filtered DataFrame size: {len(df_uttarakhand)} rows")
print(f"Final April/May-filtered DataFrame size: {len(df_april_may_unique)} rows")
