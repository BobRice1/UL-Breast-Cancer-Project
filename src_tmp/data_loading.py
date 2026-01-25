# Data loading utilities for the Breast Cancer Wisconsin datasett

import numpy as np
import pandas as pd

# Function to load and clean the Breast Cancer Wisconsin dataset
# Parameters: filepath: str - path to the .data file
# Returns: pd.DataFrame - cleaned dataframe with appropriate column names and missing rows removed


def load_breast_cancer_data(filepath: str) -> pd.DataFrame:
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

    df = pd.read_csv(filepath, header=None, names=columns)
    df = df.replace("?", np.nan)
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])
    df = df.dropna()
    return df
