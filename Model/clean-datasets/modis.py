# %%
import pandas as pd
df = pd.read_csv("../dataset/MODIS/modis_2016_India.csv")
df

# %%
df.drop(columns=['instrument', 'version'], inplace=True)
df
# %%
north_lat = 31.5
south_lat = 28.7
west_lon = 77.5
east_lon = 81.1

mask = (df['latitude'] >= south_lat) & (df['latitude'] <= north_lat) & (df['longitude'] <= east_lon) & (df['longitude'] >= west_lon)

df_uttarakhand = df[mask].copy()
df_uttarakhand
# %%
df_uttarakhand.to_csv('final-modis.csv', index=False)
