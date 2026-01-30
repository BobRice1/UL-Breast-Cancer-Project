# Unsupervised Learning approach to Breast Cancer Tumour Classification

The goal of this project is to build an understanding of the feature difference in malignant and benign breast cancer tumours, using unsupervised learning methods. For this purpose, we use the Breast Cancer Wisconsin (Original) Data Set from the UCI Machine Learning Repository.

## Structure
- `src/`
  - `data_loading.py` – Shared data loading and cleaning
  - `eda_heatmap.py` – Script to generate correlation heatmap
  - `pca_analysis.py` – PCA implementation and visualization
  - `clustering_kmeans.py` – K-Means clustering implementation
- `Data/`
  - `breast-cancer-wisconsin.data` – Raw dataset from UCI
- `Figures/`
  - Saved figures to be used in the report
- `paper/`
  - `paper.tex` - Latex Version
  - `paper.pdf` - pdf Version
 
## Data loading

All scripts import the dataset using a shared loader:

```python
from src.data_loading import load_breast_cancer_data

df = load_breast_cancer_data("Data/breast-cancer-wisconsin.data")
