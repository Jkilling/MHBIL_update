from sklearn.cluster import DBSCAN
import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score

# DBSCAN mach scikit interessant? Euclidian Distances / Density-Based Spatial Clustering of Applications with Noise

db = DBSCAN(eps=0.3, min_samples=1).fit([x, y])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot result
db = DBSCAN(eps=3, min_samples=1).fit([df_meg['x'], df_meg['y']])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters_)
mglearn.plots.plot_dbscan()

# Test approach
scaler = StandardScaler()
X_scaled = scaler.fit_transform(meg_points)

# cluster the data into clusters
dbscan = DBSCAN(eps=0.1, min_samples=1)  # eps controls the number of clusters implicitly
clusters = dbscan.fit_predict(X_scaled)

# plot the cluster assignments
plt.scatter(meg_points[:, 0], meg_points[:, 1], c=clusters, cmap="brg")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN")

# DBSCAN performance:
print("ARI =", adjusted_rand_score(y, clusters))