"""
eda.py — Data preparation pipeline extracted from ExploratoryDataAnalysis.ipynb
Used as a shared preprocessing step by both RandomForest and DecisionTree models.
"""

import numpy as np
import pandas as pd


def datapreparation(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the CustomerChurn CSV.

    Steps
    -----
    1. Drop customerID (non-predictive identifier)
    2. Fix TotalCharges (whitespace → NaN → drop rows)
    3. Binary-encode Yes/No columns
    4. Encode gender (Male=0, Female=1)
    5. Encode MultipleLines (No phone service / No → 0, Yes → 1)
    6. Encode internet-service add-ons (No internet service / No → 0, Yes → 1)
    7. Bucket tenure into 6 groups, drop raw tenure
    8. One-hot encode InternetService, Contract, PaymentMethod, tenure_group

    Parameters
    ----------
    filepath : str
        Path to CustomerChurn.csv

    Returns
    -------
    pd.DataFrame
        Model-ready feature matrix including the target column 'Churn'.
    """
    df = pd.read_csv(filepath)

    # Drop non-predictive ID
    df.drop(["customerID"], inplace=True, axis=1)

    # Fix TotalCharges: blank strings → NaN, then drop those rows
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Binary encode Yes/No columns
    binary_cols = ["Partner", "Dependents", "PaperlessBilling", "Churn", "PhoneService"]
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)

    # Gender: Male=0, Female=1
    df["gender"] = df["gender"].apply(lambda x: 0 if x == "Male" else 1)

    # MultipleLines: No phone service or No → 0, Yes → 1
    df["MultipleLines"] = df["MultipleLines"].map(
        {"No phone service": 0, "No": 0, "Yes": 1}
    )

    # Internet add-on columns: No internet service or No → 0, Yes → 1
    internet_addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in internet_addon_cols:
        df[col] = df[col].map({"No internet service": 0, "No": 0, "Yes": 1})

    # Bucket tenure into 6 groups (each representing ~12 months)
    def tenure_group(t):
        if t <= 12:   return 1
        elif t <= 24: return 2
        elif t <= 36: return 3
        elif t <= 48: return 4
        elif t <= 60: return 5
        else:         return 6

    df["tenure_group"] = df["tenure"].apply(tenure_group)
    df.drop(["tenure"], inplace=True, axis=1)

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=["InternetService", "Contract", "PaymentMethod", "tenure_group"]
    )

    return df


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python eda.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "CustomerChurn.csv"
    df = datapreparation(path)
    print("Shape :", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))