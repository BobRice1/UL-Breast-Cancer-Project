# Data analysis script for generating a heatmap of feature correlations with explanatory text.
# Data source: Breast Cancer Wisconsin dataset.

import matplotlib.pyplot as plt
import seaborn as sns

from data_loading import load_breast_cancer_data

# Light clean theme for the heatmap
plt.style.use("default")
sns.set_style("whitegrid")


df = load_breast_cancer_data("Data/breast-cancer-wisconsin.data")
update_df = df.drop(columns=["Sample_ID", "Class"])

update_df = update_df.rename(columns=lambda c: c.replace("_", " "))

# Compute correlation matrix
corr_matrix = update_df.corr(method="pearson")

# Create a figure
plt.figure(figsize=(6, 6))


# Create a heatmap of the correlation matrix
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="magma",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Pearson Correlation Matrix", pad=15)

plt.tight_layout()
plt.savefig("Figures/PearsonCorrelationMatrix.png", dpi=600, bbox_inches="tight")
plt.show()
