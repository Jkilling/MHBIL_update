import pandas as pd
import os
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from pointpats import PointPattern
import pointpats.quadrat_statistics as qs
import scipy.stats as st
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

os.getcwd()
os.chdir("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data")

# 4.1 One-Dimensional Data ===============================
crs1 = {'init': 'epsg:4326'}
df_vil_wgs84 = pd.read_excel("villages.xls")
# vil_geometry = [Point(xy) for xy in zip(df_vil_wgs84['x'], df_vil_wgs84['y'])]
# gdf_vil_84 = gpd.GeoDataFrame(df_vil_wgs84, crs=crs1, geometry=vil_geometry)
# gdf_vil_84 = gdf_vil_84.to_crs({'init': 'epsg:5677'})
vil_fd = df_vil_wgs84["AD"]

# 4.1.1 Histogram

cb = 1200, 1250, 1300, 1350, 1400
counting = range(1, 4)

np.histogram(vil_fd)

plt.hist(vil_fd, bins=6, color='lightblue', rwidth=0.95)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Time A.D.')
plt.ylabel('Frequency')
plt.title('Histogram of village foundations in different periods')

# 4.1.2 Density
# KDE
x = np.array(df_vil_wgs84['AD'])
x_d = np.linspace(1200, 1400, 201)

kde = KernelDensity(bandwidth=5, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.025)
plt.xlim(1220, 1380)
plt.xlabel('Time A.D.')
plt.ylabel('Density of village foundation')

# 4.1.3 Distance between Events Concept

plt.plot(df_vil_wgs84["AD"], df_vil_wgs84["id"], color='gray')
plt.scatter(df_vil_wgs84["AD"], df_vil_wgs84["id"], color='darkgray')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('Time A.D.')
plt.ylabel('id')

# Interval
test = np.array(df_vil_wgs84['AD'].tolist() + [df_vil_wgs84.iat[12, 2]])
test2 = np.array([df_vil_wgs84.iat[0, 2]] + (df_vil_wgs84['AD'].tolist()))

interval = (np.array(df_vil_wgs84['AD'].tolist() + [df_vil_wgs84.iat[12, 2]]) -
            np.array([df_vil_wgs84.iat[0, 2]] + (df_vil_wgs84['AD'].tolist())))

plt.plot(interval[1:12], color='gray')
# plt.scatter(interval, color='darkgray')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('Index')
plt.ylabel('Interval')

# 4.1.3 Time Series

df_meg = pd.read_csv("meg_dw.csv", sep=';')
ks_vil = pd.DataFrame({'x': x_d, 'y': np.exp(logprob)})
ks_vil.head()

# 4.2 Two-Dimensional Data ===============================

# https://nbviewer.jupyter.org/github/pysal/pointpats/blob/master/notebooks/Quadrat_statistics.ipynb

meg_points = df_meg.as_matrix(columns=df_meg.columns[1:])
pp_meg = PointPattern(meg_points)  # Window
pp_meg.plot(window=True, title="Point pattern Meg")
qr_meg = qs.QStatistic(pp_meg, shape="rectangle", nx=9, ny=6)
qr_meg.plot()

x = df_meg['x']
y = np.array(df_meg['y'])

# test
h = plt.hist2d(x, y)  # 9x6 Aufteilung
plt.colorbar(h[8])

# 4.2.1 Kernel-Based Density ===============================

# Fitting Gaussian Kernel
x = np.array(df_meg['x'])
y = np.array(df_meg['y'])

# Define the borders
xmin = min(x)
xmax = max(x)
ymin = min(y)
ymax = max(y)

print(xmin, xmax, ymin, ymax)

# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)  # Standard Rule is scott
f = np.reshape(kernel(positions).T, xx.shape)  # Als GeotTIFF mit GDAL

fig, ax = plt.subplots()
im = ax.imshow(np.rot90(f), cmap='brg', extent=[xmin, xmax, ymin, ymax])
ax.scatter(x, y, c='k', s=5, edgecolor='')

fig_2 = plt.figure()
ax = fig_2.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='brg')
ax.imshow(np.rot90(f), cmap='brg', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.ticklabel_format(style='sci')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Megaliths KDE estimation')

# KDE with Silverman

kernel_silv = st.gaussian_kde(values, bw_method='silverman')  # Silverman
f_silv = np.reshape(kernel_silv(positions).T, xx.shape)

fig_3, ax = plt.subplots()
im_3 = ax.imshow(np.rot90(f_silv), cmap='brg', extent=[xmin, xmax, ymin, ymax])
ax.scatter(x, y, c='k', s=5, edgecolor='')

# Different optic with seaorn package
ax = sns.kdeplot(x, y, shade=True, cbar=True, cmap='brg', shade_lowest=True, n_levels=350)  # n_level Anzahl spannend

# 4.2.2 Distance-Based Density ===============================

# Empty Circle Thing

# Voronoi Polygones
vor = Voronoi(meg_points)

voronoi_plot_2d(vor)
plt.title('Voronoi Polygones')
plt.xlim(xmin, xmax)
plt.xlabel('X')
plt.ylim(ymin, ymax)
plt.ylabel('Y')

#########################
#### Empty Circle Party

verti = vor.vertices

# As GPD
vert_geometry = [Point(xy) for xy in zip(verti[:, 0], verti[:, 1])]
gdf_vor = gpd.GeoDataFrame(verti, crs=crs1, geometry=vert_geometry)  # necessary for empty circle thing


# unnecessary Bounding Box Test
gdf_vor_cut = gdf_vor.cx[xmin:xmax, ymin:ymax] # Cut by bounding box
vor_cut = pd.DataFrame(gdf_vor_cut).drop(columns=['geometry'])
vor_cut.to_csv("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data/voronoi_nodes_cut.csv", index = None, header=True)

#############

# https://stackoverflow.com/questions/10650645/python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d
# https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

from scipy.spatial import cKDTree
from shapely.geometry import Point

meg_geometry = [Point(xy) for xy in zip(meg_points[:, 0], meg_points[:, 1])]
gpd_meg = gpd.GeoDataFrame(df_meg, crs=crs1, geometry=meg_geometry)
gpd_vert = gdf_vor


def ckdnearest(gdA, gdB):
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)))
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA, gdB.loc[idx, gdB.columns != 'geometry'].reset_index(),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


circle_radius = ckdnearest(gpd_vert, gpd_meg)

max(circle_radius['dist'])  # 466529.4773120758
min(circle_radius['dist'])  # 22.385581652505806

# Export as csv for further things
circle_radius.to_csv("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data/voronoi_nodes_radius.csv", index = None, header=True)

#### Furthest Site Voronoi

vertices_vor = pd.DataFrame(vor.vertices)
plt.scatter(vertices_vor[0], vertices_vor[1])
plt.title('Voronoi Polygones Vertices')
plt.xlim(xmin, xmax)
plt.xlabel('X')
plt.ylim(ymin, ymax)
plt.ylabel('Y')

vor_fur = Voronoi(meg_points, furthest_site=True) # Whether to compute a furthest-site voronoi diagram
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
voronoi_plot_2d(vor_fur)
plt.title('Voronoi Polygones with furthest site')

# 4.2.3 Decomposition ===============================

# + save image
