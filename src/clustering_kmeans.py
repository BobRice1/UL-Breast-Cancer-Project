# K-Means Clustering Implementation and Evaluation using Adjusted Rand Index (ARI)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pca_analysis import run_pca


# Compute Euclidean distance between two points (PC1, PC2)
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Centroid initialisation to k random points from the dataset as the initial centroids
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices].copy()


# Assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)


# Update centroids by calculating the mean of all points assigned to each cluster
def update_centroids(data, clusters, k, old_centroids):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]

        if len(cluster_points) == 0:
            new_centroids.append(old_centroids[i])
        else:
            new_centroids.append(cluster_points.mean(axis=0))

    return np.array(new_centroids)


# K-Means algorithm implementation - assigning clusters and updating centroids iteratively
def kmeans(data, k, max_iters=100, tol=1e-6):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)

        new_centroids = update_centroids(data, clusters, k, centroids)

        if np.allclose(centroids, new_centroids, atol=tol):
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids, clusters


def n_choose_2(n):
    return n * (n - 1) // 2


# compute the Adjusted Rand Index (ARI) between true labels and predicted cluster labels
def adjusted_rand_score(true_labels, cluster_labels):
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    n_samples = true_labels.size

    # Create contingency table
    label_classes = np.unique(true_labels)
    cluster_classes = np.unique(cluster_labels)
    contingency = np.zeros((len(label_classes), len(cluster_classes)), dtype=int)

    for i, label in enumerate(label_classes):
        for j, cluster in enumerate(cluster_classes):
            contingency[i, j] = np.sum(
                (true_labels == label) & (cluster_labels == cluster)
            )

    sum_comb_c = sum(n_choose_2(n_ij) for n_ij in contingency.flatten())
    sum_comb_rows = sum(n_choose_2(n_i) for n_i in contingency.sum(axis=1))
    sum_comb_cols = sum(n_choose_2(n_j) for n_j in contingency.sum(axis=0))
    total_combinations = n_choose_2(n_samples)

    expected_index = (sum_comb_rows * sum_comb_cols) / total_combinations
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)

    return ari


# assign k based on elbow method in pca_analysis.py, 2 optimal as 2 tumour classes present
k = 2

# Call PCA function to reduce data to 2 dimensions, y used for cluster labelling only
pca_scores_2D, y = run_pca("Data/breast-cancer-wisconsin.data", n_components=2)

# Run K-Means clustering on PC1 and PC2
centroids, clusters = kmeans(pca_scores_2D, k)

# Compute Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")

# Map cluster IDs to true labels for visualization - from 0 and 1 to Benign and Malignant
cluster_to_label = {}
for cid in np.unique(clusters):
    majority_true = np.bincount(y[clusters == cid]).argmax()
    cluster_to_label[cid] = majority_true

label_to_name = {0: "Benign", 1: "Malignant"}
cluster_names = np.array([label_to_name[cluster_to_label[c]] for c in clusters])

# Plot the clustered data with centroids
custom_palette = {"Benign": "blue", "Malignant": "red"}
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_scores_2D[:, 0],
    y=pca_scores_2D[:, 1],
    hue=cluster_names,
    palette=custom_palette,
)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c="black", marker="X")
plt.title("K-Means Clustering on PCA-Reduced Data", fontsize=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("Figures/K-means_pca.png", dpi=300, bbox_inches="tight")
plt.show()
