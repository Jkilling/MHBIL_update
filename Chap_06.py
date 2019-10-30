import os
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

os.getcwd()
os.chdir("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data")

# Data
sgdf_srtm = rasterio.open(r'DEM.tiff', crs={'init': 'epsg:5677'})
crs = {'init': 'epsg:5677'}
df_meg = pd.read_csv("meg_dw.csv", sep=';')
gdf_meg = gpd.GeoDataFrame(df_meg, crs=crs, geometry = meg_geometry)

# Build / Transform Raster Object

# Plot of raster with locations
