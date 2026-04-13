"""
random_forest.py — Random Forest model extracted from RandomForest__1_.ipynb

Workflow
--------
1. train_and_save(filepath, model_path)  — trains the best RF and saves it as .pkl
2. load_model(model_path)                — loads a saved model
3. predict(input_dict, model_path)       — returns prediction + probability for Flask
4. get_metrics(model_path, filepath)     — returns evaluation metrics dict for Flask

CLI usage
---------
    python random_forest.py --train CustomerChurn.csv
    python random_forest.py --predict '{"MonthlyCharges": 70.5, ...}'
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from imblearn.combine import SMOTEENN

from eda import datapreparation

# Default path where the trained model is saved
DEFAULT_MODEL_PATH = "saved_models/random_forest.pkl"
DEFAULT_COLUMNS_PATH = "saved_models/rf_columns.pkl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_dataset(filepath: str):
    """Return X, y, X_train, X_test, y_train, y_test (SMOTEENN-resampled train)."""
    df = datapreparation(filepath)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote_enn = SMOTEENN(random_state=42)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    return X, y, X_train, X_test, y_train, y_test


def _select_best(models: dict, X_test, y_test) -> tuple:
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall":    round(recall_score(y_test, preds), 4),
            "AUC-ROC":   round(roc_auc_score(y_test, probs), 4),
        })

    results = pd.DataFrame(rows)
    print("\n--- Model Comparison ---")
    print(results.to_string(index=False))

    best_row  = results.sort_values(["Recall", "AUC-ROC"], ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\nSelected: {best_name}  |  Recall={best_row['Recall']}  |  AUC-ROC={best_row['AUC-ROC']}")
    return best_name, models[best_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_save(
    filepath: str = "customer_churn.csv",
    model_path: str = DEFAULT_MODEL_PATH,
    columns_path: str = DEFAULT_COLUMNS_PATH,
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y, X_train, X_test, y_train, y_test = _build_dataset(filepath)

    # --- Stage 1: default model ---
    print("Fitting default RF …")
    rf1 = RandomForestClassifier(random_state=42)
    rf1.fit(X_train, y_train)

    # --- Stage 2: tune n_estimators & max_features ---
    print("Grid search 1/3 (n_estimators, max_features) …")
    grid1 = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [300, 500, 700, 1000], "max_features": ["sqrt", "log2", None]},
        cv=3, scoring="recall", n_jobs=-1, verbose=1,
    )
    grid1.fit(X_train, y_train)
    rf2 = grid1.best_estimator_
    print("Best (grid 1):", grid1.best_params_)

    # --- Stage 3: tune criterion, max_depth, class_weight ---
    print("Grid search 2/3 (criterion, max_depth, class_weight) …")
    grid2 = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators":  [grid1.best_params_["n_estimators"]],
            "max_features":  [grid1.best_params_["max_features"]],
            "criterion":     ["gini", "entropy"],
            "max_depth":     [10, 20, 30, None],
            "class_weight":  [{0: 1, 1: 2}, {0: 1, 1: 3}],
        },
        cv=3, scoring="recall", n_jobs=-1, verbose=1,
    )
    grid2.fit(X_train, y_train)
    rf3 = grid2.best_estimator_
    print("Best (grid 2):", grid2.best_params_)

    # --- Stage 4: tune min_samples_split & min_samples_leaf ---
    print("Grid search 3/3 (min_samples_split, min_samples_leaf) …")
    grid3 = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators":      [grid1.best_params_["n_estimators"]],
            "max_features":      [grid1.best_params_["max_features"]],
            "criterion":         [grid2.best_params_["criterion"]],
            "max_depth":         [grid2.best_params_["max_depth"]],
            "class_weight":      [grid2.best_params_["class_weight"]],
            "min_samples_split": [2, 4, 8, 10],
            "min_samples_leaf":  [1, 3, 5, 7],
        },
        cv=3, scoring="recall", n_jobs=-1, verbose=1,
    )
    grid3.fit(X_train, y_train)
    rf4 = grid3.best_estimator_
    print("Best (grid 3):", grid3.best_params_)

    # --- Select best overall model ---
    _, final_model = _select_best(
        {"RF Default": rf1, "RF Grid1": rf2, "RF Grid2": rf3, "RF Fully Tuned": rf4},
        X_test, y_test,
    )

    # --- Save model + column list ---
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    with open(columns_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\nModel saved to  : {model_path}")
    print(f"Columns saved to: {columns_path}")
    return final_model


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and return the saved RandomForest model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(
    input_data: dict,
    model_path: str = DEFAULT_MODEL_PATH,
    columns_path: str = DEFAULT_COLUMNS_PATH,
) -> dict:
    """
    Run inference for a single customer record.

    Parameters
    ----------
    input_data   : dict of feature_name → value  (already preprocessed / encoded)
    model_path   : path to the saved .pkl model
    columns_path : path to the saved column list .pkl

    Returns
    -------
    dict with keys:
        prediction  : 0 (No Churn) or 1 (Churn)
        probability : churn probability (float, 0–1)
        label       : human-readable label string
    """
    model = load_model(model_path)

    with open(columns_path, "rb") as f:
        columns = pickle.load(f)

    # Build a single-row DataFrame aligned to training columns
    row = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)

    pred  = int(model.predict(row)[0])
    prob  = float(model.predict_proba(row)[0][1])

    return {
        "prediction":  pred,
        "probability": round(prob, 4),
        "label":       "Churn" if pred == 1 else "No Churn",
    }


def get_metrics(
    filepath: str = "CustomerChurn.csv",
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Evaluate the saved model and return a metrics dict suitable for a Flask response.

    Returns
    -------
    dict with accuracy, precision, recall, auc_roc,
         confusion_matrix, classification_report
    """
    model = load_model(model_path)
    _, _, _, X_test, _, y_test = _build_dataset(filepath)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":               round(accuracy_score(y_test, y_pred), 4),
        "precision":              round(precision_score(y_test, y_pred), 4),
        "recall":                 round(recall_score(y_test, y_pred), 4),
        "auc_roc":                round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"]
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest — Customer Churn")
    parser.add_argument("--train",   metavar="CSV",  help="Train and save model from CSV")
    parser.add_argument("--predict", metavar="JSON", help="Predict from JSON dict string")
    parser.add_argument("--metrics", metavar="CSV",  help="Print metrics for saved model")
    args = parser.parse_args()

    if args.train:
        train_and_save(filepath=args.train)

    elif args.predict:
        result = predict(json.loads(args.predict))
        print(json.dumps(result, indent=2))

    elif args.metrics:
        metrics = get_metrics(filepath=args.metrics)
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"{k:20s}: {v}")
        print("\nConfusion Matrix:\n", np.array(metrics["confusion_matrix"]))
        print("\nClassification Report:\n", metrics["classification_report"])

    else:
        parser.print_help()