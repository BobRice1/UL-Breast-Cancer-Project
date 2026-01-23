# Unsupervised Learning approach to Breast Cancer Tumour Classification

The goal of this project is to build an understanding of the feature difference in malignant and benign tumours, using unsupervised learning methods. For this purpose, we use the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository.

## Structure
- `src/`
  - `data_loading.py` – shared data loading and cleaning
- `Data/`
  - `breast-cancer-wisconsin.data` – raw dataset from UCI
- `Figures/`
  - Saved figures to be used in the report
- `paper/`
  - to be added
 
## Data loading

All scripts import the dataset using a shared loader:

```python
from src.data_loading import load_breast_cancer_data

df = load_breast_cancer_data("Data/breast-cancer-wisconsin.data")
