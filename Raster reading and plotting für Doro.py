import rasterio
import os

os.getcwd()
os.chdir("C:/Users/johan/OneDrive/Desktop/data/raw_data")  # If path doesn't exist : FileNotFoundError

elev_perg = rasterio.open(r'TUR_msk_alt.vrt', crs={'init': 'epsg:4326'})

base = show(elev_perg, 1, cmap='terrain')