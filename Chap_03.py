# importing packages
from shapely.geometry import Point  # Shapely for converting latitude/longitude to geometry
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import rasterio
from rasterio.plot import show
import shapely.speedups

# 2.3. Starting with a script  ===============================
# Setting working directory
os.getcwd()
os.chdir("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data")  # If path doesn't exist : FileNotFoundError

# ------ Filing / CRS
file_meg = "meg_dw.csv"
file_tum = "tum_dw.csv"
file_coast1 = "coast_gk3.shp"
file_coast = "coast_gk3"
file_srtm = "dw_gk3_50_ag.asc"
file_vil = "villages.xls"

crs = {'init': 'epsg:5677'}  # GK
crs1 = {'init': 'epsg:4326'}  # WGS84

# ------
# Load data csv
# df meg
df_meg = pd.read_csv("meg_dw.csv", sep=';')
# spatialize
# creating a geometry column
meg_geometry = [Point(xy) for xy in zip(df_meg['x'], df_meg['y'])]
# Creating a Geographic data frame
gdf_meg = gpd.GeoDataFrame(df_meg, crs=crs, geometry=meg_geometry)
# gdf_meg.plot(c='red')


# df tum
df_tum = pd.read_csv(file_tum, sep=';')
# spatialize
# creating a geometry column
tum_geometry = [Point(xy) for xy in zip(df_tum['x'], df_tum['y'])]
gdf_tum = gpd.GeoDataFrame(df_tum, crs=crs, geometry=tum_geometry)
# gdf_tum.plot(c='blue')

# ------
# Load data SHP
coast1 = gpd.read_file('coast_gk3.shp')
# coast1.plot()

# ------
# Load data Excel
df_vil_wgs84 = pd.read_excel(file_vil)
vil_geometry = [Point(xy) for xy in zip(df_vil_wgs84['x'], df_vil_wgs84['y'])]
gdf_vil_84 = gpd.GeoDataFrame(df_vil_wgs84, crs=crs1, geometry=vil_geometry)

# gdf_vil_84.plot(c='orange')

# access coordinates in spatial objects.

df_vil_wgs84['y']
df_vil_wgs84['x']

# ------
# Load data ASCII
gdal.AllRegister()
driver = gdal.GetDriverByName("GTiff")
# Set the reference info
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS("DHDN / 3-degree Gauss-Kruger zone 3")
srs.ImportFromEPSG(5677)

ds = gdal.Open("dw_gk3_50_ag.asc")
dst_ds = driver.CreateCopy("DEM.tiff", ds)
dst_ds.SetProjection(srs.ExportToWkt())
del dst_ds

# ------
# Reproject village points
villages = gdf_vil_84.to_crs({'init': 'epsg:5677'})
# villages.plot()

# ------
# Bounding Box
# Create Bounding Layer
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()
sgdf_gdal = gdal.Open(r'DEM.tiff')

# Polygonize for further Usage
shapely.speedups.enable()

srcband = sgdf_gdal.GetRasterBand(1)
dst_layername = "Bound_raster"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(dst_layername + ".shp")
dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
gdal.Polygonize(srcband, None, dst_layer, -1, callback=None)

# Read bounding Shape
bb_layer = gpd.read_file('Bound_raster.shp', driver='ESRI Shapefile')
bb_layer.head()

vil_mask = villages.within(bb_layer.loc[0, 'geometry'])
print(vil_mask)
pip_vil = bb_layer.loc[vil_mask]  # Läuft noch nicht

# ------
# Remove duplicates currently no real elegant solution / pandas doesn't work

# convert to wkb
gdf_vil_84["geometry"] = gdf_vil_84["geometry"].apply(lambda geom: geom.wkb)

df = gdf_vil_84.drop_duplicates(["geometry"])

# convert back to shapely geometry
gdf_vil_84["geometry"] = df["geometry"].apply(lambda geom: shapely.wkb.loads(geom))


# ------
# Slope
def calculate_slope(DEM):
    gdal.DEMProcessing('slope.tif', DEM, 'slope')
    with rasterio.open('slope.tif') as dataset:
        slope = dataset.read(1)
    return slope


calculate_slope('DEM.tiff')

slope_srtm = rasterio.open(r'slope.tif', crs={'init': 'epsg:5677'})
slope_plot = show(slope_srtm, 1)


# Aspect
def calculate_aspect(DEM):
    gdal.DEMProcessing('aspect.tif', DEM, 'aspect')
    with rasterio.open('aspect.tif') as dataset:
        aspect = dataset.read(1)


calculate_aspect('DEM.tiff')

aspect_srtm = rasterio.open(r'aspect.tif', crs={'init': 'epsg:5677'})
aspect_plot = show(aspect_srtm, 1)

# ------
# Plotting

sgdf_srtm = rasterio.open(r'DEM.tiff', crs={'init': 'epsg:5677'})
base = show(sgdf_srtm, cmap='terrain')
villages.plot(ax=base, color='black', markersize=16, marker="s")
gdf_tum.plot(ax=base, c='red', markersize=16, marker="o")
gdf_meg.plot(ax=base, c='blue', markersize=16, marker="^")
blue_patch = mlines.Line2D([], [], color='blue', markersize=7, marker="^",
                           linestyle='None', label='Neolithic Megaliths')
red_patch = mlines.Line2D([], [], color='red', markersize=7, marker="o",
                          linestyle='None', label='Bronze Age Barrows')
black_patch = mlines.Line2D([], [], color='black', markersize=7, marker="s",
                            linestyle='None', label='Mediaeval Villages')

plt.legend(handles=[red_patch, blue_patch, black_patch], loc='lower right')
plt.text(3562000, 6030000, "Dänischer Wohld", transform=base.transData)
plt.text(3565000, 6039600, "Baltic Sea", transform=base.transData)

# ------
# Plotting 2.0

# https://ocefpaf.github.io/python4oceanographers/blog/2015/03/02/geotiff/

with rasterio.open('DEM.tiff') as src:
    subset = src.read(1)

plt.figure(figsize=(6, 8.5))
plt.imshow(subset, cmap='terrain')
plt.colorbar(shrink=0.5)

# 2.4. Helpful functions ===============================
a = 5
b = 0
c = 100
i = 0

# for loop
for x in range(5):  # range function starts at 0
    print(x)

# while loop
while i < c:
    b = b + 1
    i = i + a

print(b)
print(i)

# DataFrame from scratch
ID = 1, 2, 3, 4, 5, 6
diameter = 3, 6, 4, 4, 2, 9
length = 23, 32, 12, 22, 16, 77
colour = "red", "red", "blue", "red", "blue", "green"

data = pd.DataFrame({'ID': ID,
                     'diameter': diameter,
                     'lengtth': length,
                     'colour': colour})

data.rename(columns={'lengtth': 'length'}, inplace=True)