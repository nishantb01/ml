"""Experiment 8: select a K-Means cluster count with inertia and silhouette."""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = load_iris().data
inertias = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(model.inertia_)
    print(f"K={k}, inertia={model.inertia_:.3f}, silhouette={silhouette_score(X, model.labels_):.3f}")

