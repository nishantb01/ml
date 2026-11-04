"""Experiment 9: DBSCAN clustering and noise detection."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

X = load_iris().data[:, :2]
X_scaled = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print("Cluster labels:\n", labels)
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("DBSCAN Clustering")
plt.show()

