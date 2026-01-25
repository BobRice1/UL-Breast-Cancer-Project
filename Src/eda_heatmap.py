# Data analysis script for generating a heatmap of feature correlations with explanatory text.
# Data source: Breast Cancer Wisconsin dataset.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loading import load_breast_cancer_data

# Light clean theme for the heatmap
plt.style.use("default")
sns.set_style("whitegrid")

# Load and clean data
columns = [
    "Sample_ID",
    "Clump_Thickness",
    "Uniformity_Cell_Size",
    "Uniformity_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
    "Class",
]

df = load_breast_cancer_data("Data/breast-cancer-wisconsin.data")
update_df = df.drop(columns=["Sample_ID", "Class"])

# Compute correlation matrix
corr_matrix = update_df.corr()

# Create figure with two panels: left for text, right for heatmap
plt.figure(figsize=(7, 6))

# Heatmap - (Pearson correlation matrix)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="magma",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("Figures/FeatureCorrelationHeatmap.png", dpi=300, bbox_inches="tight")
plt.show()
