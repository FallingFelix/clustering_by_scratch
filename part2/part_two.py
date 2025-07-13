import matplotlib
matplotlib.use("TkAgg")  # Use GUI backend

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = np.loadtxt("510_cluster_dataset.txt")

def run_kmeans(data, k, r=10):
    best_sse = float("inf")
    best_kmeans = None
    for _ in range(r):
        kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=300)
        kmeans.fit(data)
        sse = kmeans.inertia_
        if sse < best_sse:
            best_sse = sse
            best_kmeans = kmeans
    return best_kmeans, best_sse

for k in [2, 3, 4]:
    model, sse = run_kmeans(data, k)
    labels = model.labels_
    centers = model.cluster_centers_

    print(f"K={k}, SSE={sse:.2f}")

    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.title(f"K-Means Clustering (K={k})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"kmeans_k{k}.png")
    plt.show()  # Opens a desktop window

