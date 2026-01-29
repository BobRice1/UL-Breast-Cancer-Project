# K-Means Clustering Implementation and Evaluation using Adjusted Rand Index (ARI)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pca_analysis import run_pca


# Compute squared Euclidean distances between data points and centroids
def _squared_distances(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:

    # Returns a (n_samples, n_centroids) array of squared distances
    return ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


# Initialize centroids by randomly selecting k unique data points
def initialize_centroids_random(
    data: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:

    indices = rng.choice(data.shape[0], size=k, replace=False)
    return data[indices].copy()


# Initialize centroids using k-means++ method
def initialize_centroids_kmeanspp(
    data: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """
    - Choose first centroid uniformly at random.
    - Choose subsequent centroids with probability proportional to distance^2 to
      the closest existing centroid.
    """
    # Initialize centroids array
    n_samples = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=float)

    # Choose the first centroid randomly
    first_idx = rng.integers(0, n_samples)
    centroids[0] = data[first_idx]

    # Choose remaining centroids
    closest_d2 = ((data - centroids[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = closest_d2.sum()
        if total <= 0:
            # All points identical (or numerical collapse): fall back to random picks
            remaining = rng.choice(n_samples, size=(k - c), replace=False)
            centroids[c:] = data[remaining]
            break

        probs = closest_d2 / total
        next_idx = rng.choice(n_samples, p=probs)
        centroids[c] = data[next_idx]
        next_d2 = ((data - centroids[c]) ** 2).sum(axis=1)
        closest_d2 = np.minimum(closest_d2, next_d2)

    return centroids


# Update centroids by calculating the mean of all points assigned to each cluster
def update_centroids(
    data: np.ndarray,
    clusters: np.ndarray,
    k: int,
    old_centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new_centroids = np.empty_like(old_centroids, dtype=float)

    for i in range(k):
        cluster_points = data[clusters == i]
        if cluster_points.shape[0] == 0:
            # Re-seed empty cluster to the point with largest current error
            d2 = ((data - old_centroids[clusters]) ** 2).sum(axis=1)
            new_centroids[i] = data[np.argmax(d2)]
        else:
            new_centroids[i] = cluster_points.mean(axis=0)

    return new_centroids


# Run a single k-means clustering
def kmeans_single_run(
    data: np.ndarray,
    k: int,
    *,
    max_iters: int = 300,
    tol: float = 1e-6,
    init: str = "kmeans++",
    rng: np.random.Generator,
):
    if init == "kmeans++":
        centroids = initialize_centroids_kmeanspp(data, k, rng)
    elif init == "random":
        centroids = initialize_centroids_random(data, k, rng)
    else:
        raise ValueError("init must be 'kmeans++' or 'random'")

    # K-means main loop
    prev_inertia = None
    for it in range(1, max_iters + 1):
        d2 = _squared_distances(data, centroids)
        clusters = np.argmin(d2, axis=1)
        inertia = float(np.min(d2, axis=1).sum())

        centroids = update_centroids(data, clusters, k, centroids, rng)

        if prev_inertia is not None:
            rel_improvement = abs(prev_inertia - inertia) / max(prev_inertia, 1e-12)
            if rel_improvement < tol:
                break
        prev_inertia = inertia

    return centroids, clusters, inertia, it


# Run k-means with multiple random initialisations (restarts).
def kmeans(
    data: np.ndarray,
    k: int,
    *,
    n_init: int = 20,
    random_state: int = 0,
    init: str = "kmeans++",
    max_iters: int = 300,
    tol: float = 1e-6,
):
    # Set up random number generator
    rng = np.random.default_rng(random_state)

    # Keep track of best solution found
    best = None
    for _ in range(n_init):
        centroids, clusters, inertia, n_iters = kmeans_single_run(
            data,
            k,
            max_iters=max_iters,
            tol=tol,
            init=init,
            rng=rng,
        )
        if best is None or inertia < best["inertia"]:
            best = {
                "centroids": centroids,
                "clusters": clusters,
                "inertia": inertia,
                "n_iters": n_iters,
            }

    # Return best solution found with lowest inertia
    return best["centroids"], best["clusters"], best["inertia"], best["n_iters"]


# Helper function to compute n choose 2
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

# k-means configuration (for reproducibility + robustness)
RANDOM_STATE = 0
N_INIT = 30
INIT_METHOD = "kmeans++"  # 'kmeans++' or 'random'

# ARI over multiple PCA component counts
component_list = [2, 3, 5, 10]

ari_results = {}
best = {
    "n_components": None,
    "ari": -np.inf,
    "centroids": None,
    "clusters": None,
    "scores": None,
    "inertia": None,
    "n_iters": None,
}

for n_comp in component_list:
    pca_scores, y = run_pca("Data/breast-cancer-wisconsin.data", n_components=n_comp)

    centroids, clusters, inertia, n_iters = kmeans(
        pca_scores,
        k,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
        init=INIT_METHOD,
    )

    ari = adjusted_rand_score(y, clusters)
    ari_results[n_comp] = ari

    print(
        f"n_components={n_comp:>2} ARI={ari:.3f} inertia={inertia:.1f} iters={n_iters}"
    )
    if ari > best["ari"]:
        best.update(
            {
                "n_components": n_comp,
                "ari": ari,
                "centroids": centroids,
                "clusters": clusters,
                "scores": pca_scores,
                "inertia": inertia,
                "n_iters": n_iters,
            }
        )

print(f"\nBest ARI={best['ari']:.3f} at n_components={best['n_components']}")

# Use the best result for plotting
pca_scores_best = best["scores"]
clusters_best = best["clusters"]
centroids_best = best["centroids"]

# Map cluster IDs to true labels for visualization - from 0 and 1 to Benign and Malignant
cluster_to_label = {}
for cid in np.unique(clusters_best):
    majority_true = np.bincount(y[clusters_best == cid]).argmax()
    cluster_to_label[cid] = majority_true

label_to_name = {0: "Benign", 1: "Malignant"}
cluster_names = np.array([label_to_name[cluster_to_label[c]] for c in clusters_best])

# Plot the clustered data with centroids
custom_palette = {"Benign": "blue", "Malignant": "red"}
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_scores_best[:, 0],
    y=pca_scores_best[:, 1],
    hue=cluster_names,
    palette=custom_palette,
)
plt.scatter(centroids_best[:, 0], centroids_best[:, 1], s=300, c="black", marker="X")
plt.title("K-Means Clustering on PCA-Reduced Data", fontsize=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("Figures/K-means_pca.png", dpi=300, bbox_inches="tight")
plt.show()
