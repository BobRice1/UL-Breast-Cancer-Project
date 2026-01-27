# PCA analysis script for Breast Cancer Wisconsin Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # just for type hints / convenience

# Import the shared data loading function
from data_loading import load_breast_cancer_data  # adjust import path if needed


# Standardise features to zero mean and unit variance
def standardise_features(feature_matrix: np.ndarray) -> np.ndarray:
    # Standardisation: (X - mean) / std
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0, ddof=0)  # population std; ddof=1 also fine
    # Avoid division by zero just in case
    stds[stds == 0] = 1.0
    standardised_features = (feature_matrix - means) / stds
    return standardised_features


# Compute PCA via NumPy eigen-decomposition of the covariance matrix
def compute_pca(X_std: np.ndarray):

    # Covariance matrix of features - features in columns
    cov_matrix = np.cov(X_std, rowvar=False)

    # Eigen-decomposition; eigh is used for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues/eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sort_indices]
    eigenvectors_sorted = eigenvectors[:, sort_indices]

    return eigenvalues_sorted, eigenvectors_sorted


# Project standardised data onto first n principal components
def project_onto_pcs(
    standardised_features: np.ndarray,
    eigenvectors_sorted: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:

    # Select the first n_components eigenvectors
    projection_matrix = eigenvectors_sorted[:, :n_components]

    # Project the data
    pca_scores = standardised_features @ projection_matrix

    return pca_scores


# Compute how much each feature contributes to each principal component
def compute_loadings_df(
    eigenvectors_sorted: np.ndarray,
    feature_names: list[str],
    n_components: int = 2,
) -> pd.DataFrame:
    # Select the first n_components eigenvectors
    loadings = eigenvectors_sorted[:, :n_components]

    # Create a DataFrame for better readability
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return loadings_df


# Print top positive and negative loadings (most important features) for a given principal component
def print_loadings(loadings_df: pd.DataFrame, pc: str = "PC1", top_n: int = 5) -> None:
    # Sort loadings by absolute value for the specified principal component
    sorted_loadings = loadings_df[pc]
    # Get top positive and negative contributors
    top_positive = sorted_loadings.sort_values(ascending=False).head(top_n)
    top_negative = sorted_loadings.sort_values(ascending=True).head(top_n)

    print(f"\nTop + loadings for {pc}:")
    print(top_positive.to_string())

    print(f"\nTop - loadings for {pc}:")
    print(top_negative.to_string())


# Bar plot to visualise most important feature loadings for a given principal component
def plot_loadings_bar(
    loadings_df: pd.DataFrame,
    pc: str = "PC1",
    top_n: int = 8,
    output_path: str = "Figures/pca_loadings_pc1.png",
) -> None:

    # Select loadings for the specified principal component
    s = loadings_df[pc].copy()

    # Get top_n features by absolute loading value
    top_features = s.abs().sort_values(ascending=False).head(top_n).index
    # Select and sort these features for plotting
    s = s.loc[top_features].sort_values()

    plt.figure(figsize=(6, 4))
    plt.barh(s.index, s.values, alpha=0.8)  # horizontal bar plot
    plt.title(f"Feature Loadings for {pc}")
    plt.xlabel("Loading Value")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# Plot with explained variance in percent and red lines marking cumulative variance at selected components.
def plot_prob_of_variance(
    eigenvalues_sorted: np.ndarray,
    highlight_components=(2, 3),
    output_path: str = "Figures/pca_variance.png",
) -> None:

    # Calculate explained variance ratios
    total_var = eigenvalues_sorted.sum()
    explained_variance_ratio = eigenvalues_sorted / total_var

    # Cumulative explained variance
    cumulative = np.cumsum(explained_variance_ratio)

    # Number of components
    n_components = len(eigenvalues_sorted)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Individual variance (bars) in percentage
    ax.bar(
        range(1, n_components + 1),
        explained_variance_ratio * 100,
        alpha=0.7,
        label="Individual variance",
    )

    # Cumulative variance (line) in percentage
    ax.plot(
        range(1, n_components + 1),
        cumulative * 100,
        marker="o",
        linestyle="--",
        label="Cumulative variance",
    )

    # Red lines for highlighted components (e.g. after PC2 and PC3)
    for k in highlight_components:
        if 1 <= k <= n_components:
            y_val = cumulative[k - 1] * 100

        # Horizontal threshold line
        ax.axhline(y=y_val, color="red", linestyle=":", linewidth=1.2)

        # Label outside the plot area (to the right)
        ax.annotate(
            f"{k} PCs: {y_val:.1f}%",
            xy=(1.02, y_val),
            xycoords=("axes fraction", "data"),
            color="red",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Proportion of Variance")
    ax.set_xticks(range(1, n_components + 1))
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# Scatter plot of PC1 vs PC2 coloured by tumour class - benign vs malignant
def plot_pca_scatter(
    pca_scores: np.ndarray, y: np.ndarray, output_path: str = "Figures/pca_scatter.png"
) -> None:

    plt.figure(figsize=(6, 5))
    benign_mask = y == 0
    malignant_mask = y == 1

    plt.scatter(
        pca_scores[benign_mask, 0],
        pca_scores[benign_mask, 1],
        label="Benign",
        color="blue",
        alpha=0.7,
        marker="x",
    )
    plt.scatter(
        pca_scores[malignant_mask, 0],
        pca_scores[malignant_mask, 1],
        label="Malignant",
        color="red",
        alpha=0.7,
        marker="x",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Principal Component Analysis: PC1 vs PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# Scree plot of eigenvalues with elbow marked at PC2
def plot_scree(
    eigenvalues_sorted: np.ndarray, output_path: str = "Figures/pca_scree.png"
) -> None:

    n_components = len(eigenvalues_sorted)

    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, n_components + 1),
        eigenvalues_sorted,
        marker="o",
        linestyle="--",
        label="Eigenvalues",
    )
    plt.xlabel("Components")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    plt.xticks(range(1, n_components + 1))

    # Mark the elbow point at PC2
    elbow_index = 2
    plt.axvline(
        x=elbow_index,
        color="red",
        linestyle=":",
        linewidth=1.2,
    )
    plt.text(
        elbow_index + 0.1,
        eigenvalues_sorted[elbow_index - 1],
        "Elbow",
        color="red",
        va="center",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    # Load data
    df = load_breast_cancer_data("Data/breast-cancer-wisconsin.data")

    # Build feature matrix X and label vector y
    feature_df: pd.DataFrame = df.drop(columns=["Sample_ID", "Class"])
    feature_matrix = feature_df.values.astype(float)
    target_labels = df["Class"].map({2: 0, 4: 1}).values  # 0 = benign, 1 = malignant

    # Standardise features
    features_std = standardise_features(feature_matrix)

    # PCA via eigen-decomposition
    eigenvalues_sorted, eigenvectors_sorted = compute_pca(features_std)

    # Variance plot (all components)
    plot_prob_of_variance(eigenvalues_sorted, highlight_components=(2, 3))

    # Project onto first two PCs and scatter plot
    pca_scores_2d = project_onto_pcs(features_std, eigenvectors_sorted, n_components=2)
    plot_pca_scatter(pca_scores_2d, target_labels)

    # Scree plot
    plot_scree(eigenvalues_sorted)

    # Compute and display feature loadings
    feature_names = feature_df.columns.tolist()
    loadings_df = compute_loadings_df(
        eigenvectors_sorted, feature_names, n_components=2
    )

    print_loadings(loadings_df, pc="PC1", top_n=5)
    print_loadings(loadings_df, pc="PC2", top_n=5)

    plot_loadings_bar(
        loadings_df, pc="PC1", top_n=8, output_path="Figures/pca_loadings_pc1.png"
    )
    plot_loadings_bar(
        loadings_df, pc="PC2", top_n=8, output_path="Figures/pca_loadings_pc2.png"
    )


if __name__ == "__main__":
    main()

# Helper function - used in clustering script


def run_pca(filepath: str, n_components: int = 2):

    # Load data
    df = load_breast_cancer_data(filepath)

    # Remove non-feature columns, convert to NumPy arrays, shape (n_samples, n_features)
    update_df = df.drop(columns=["Sample_ID", "Class"])
    features = update_df.values.astype(float)

    # Get labels
    y = df["Class"].map({2: 0, 4: 1}).values

    # Standardise and run PCA
    standardised = standardise_features(features)

    _, eigenvectors_sorted = compute_pca(standardised)

    # Multiply standardised data by first n_components eigenvectors
    pca_scores = project_onto_pcs(standardised, eigenvectors_sorted, n_components)

    return pca_scores, y
