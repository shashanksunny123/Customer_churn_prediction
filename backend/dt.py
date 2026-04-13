
import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection  import train_test_split, GridSearchCV
from sklearn.tree             import DecisionTreeClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from imblearn.combine import SMOTEENN

from eda import datapreparation

DEFAULT_MODEL_PATH   = "saved_models/decision_tree.pkl"
DEFAULT_COLUMNS_PATH = "saved_models/dt_columns.pkl"


# ── Dataset builder ──────────────────────────────────────────────
def _build_dataset(filepath: str):
    """Split + SMOTEENN resample, exactly as in notebook."""
    df = datapreparation(filepath)
    X  = df.drop(columns=["Churn"])
    y  = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote_enn = SMOTEENN(random_state=42)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    return X, y, X_train, X_test, y_train, y_test


# ── Train & save ─────────────────────────────────────────────────
def train_and_save(
    filepath:     str = "customer_churn.csv",
    model_path:   str = DEFAULT_MODEL_PATH,
    columns_path: str = DEFAULT_COLUMNS_PATH,
):
    """
    Exact replication of DecisionTree.ipynb +
    CalibratedClassifierCV to fix 0/1 probability issue.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y, X_train, X_test, y_train, y_test = _build_dataset(filepath)

    # ── Cell 14: Default DT ──────────────────────────────────────
    print("Fitting default Decision Tree …")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # ── Cell 15: Default metrics ─────────────────────────────────
    y_pred_base = dt.predict(X_test)
    print("\n── Default DT ──")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred_base):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred_base):.4f}")
    print(classification_report(y_test, y_pred_base, target_names=["No Churn", "Churn"]))
    print(f"Tree Depth: {dt.get_depth()}")

    # ── Cell 18: GridSearchCV ────────────────────────────────────
    print("\nRunning GridSearchCV …")
    grid_params = {
        "criterion":         ["gini", "entropy"],
        "max_depth":         [10, 20, 30, None],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf":  [1, 4, 8],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        grid_params,
        cv=3,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)

    # ── Cell 20: best_dt ─────────────────────────────────────────
    best_dt     = grid.best_estimator_
    y_pred_best = best_dt.predict(X_test)

    # ── Cell 21: Tuned metrics ───────────────────────────────────
    print("\n── Tuned DT (best_dt) ──")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred_best):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred_best):.4f}")
    print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))
    print(f"Tree Depth: {best_dt.get_depth()}")

    # ── FIX: Calibrate probabilities ─────────────────────────────
    # Raw DT leaf nodes are 100% pure after SMOTEENN → hard 0/1 probabilities.
    # CalibratedClassifierCV (isotonic, cv=3) learns a smooth
    # monotonic mapping: raw_score → calibrated_probability.
    # Predictions (0/1) are unchanged; only predict_proba() is smoothed.
    print("\nCalibrating probabilities (isotonic regression) …")
    calibrated_dt = CalibratedClassifierCV(
        estimator=best_dt,
        method="isotonic",   # isotonic > sigmoid for DT (non-parametric)
        cv=3,
    )
    calibrated_dt.fit(X_train, y_train)

    # Verify calibration fixed the issue
    raw_probs  = best_dt.predict_proba(X_test)[:, 1]
    cal_probs  = calibrated_dt.predict_proba(X_test)[:, 1]
    unique_raw = len(np.unique(np.round(raw_probs, 2)))
    unique_cal = len(np.unique(np.round(cal_probs, 2)))
    print(f"Unique probability values — Raw: {unique_raw}  |  Calibrated: {unique_cal}")
    print(f"Raw prob range    : [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
    print(f"Calibrated range  : [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

    # Accuracy unchanged after calibration (predictions same)
    y_pred_cal = calibrated_dt.predict(X_test)
    print(f"\nAccuracy after calibration: {accuracy_score(y_test, y_pred_cal):.4f}")

    # ── Save calibrated model ─────────────────────────────────────
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_dt, f)
    with open(columns_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\nCalibrated model saved → {model_path}")
    print(f"Columns saved         → {columns_path}")
    return calibrated_dt


# ── Load ─────────────────────────────────────────────────────────
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── Predict ──────────────────────────────────────────────────────
def predict(
    input_data:   dict,
    model_path:   str = DEFAULT_MODEL_PATH,
    columns_path: str = DEFAULT_COLUMNS_PATH,
) -> dict:
    """Single-record inference with smooth calibrated probabilities."""
    model = load_model(model_path)
    with open(columns_path, "rb") as f:
        columns = pickle.load(f)

    row  = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    return {
        "prediction":  pred,
        "probability": round(prob, 4),
        "label":       "Churn" if pred == 1 else "No Churn",
    }


# ── Metrics ──────────────────────────────────────────────────────
def get_metrics(
    filepath:   str = "CustomerChurn.csv",
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    model = load_model(model_path)
    _, _, _, X_test, _, y_test = _build_dataset(filepath)
    y_pred = model.predict(X_test)

    return {
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "precision":             round(precision_score(y_test, y_pred), 4),
        "recall":                round(recall_score(y_test, y_pred), 4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"]
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree — Customer Churn")
    parser.add_argument("--train",   metavar="CSV",  help="Train and save model")
    parser.add_argument("--predict", metavar="JSON", help="Predict from JSON string")
    parser.add_argument("--metrics", metavar="CSV",  help="Print evaluation metrics")
    args = parser.parse_args()

    if args.train:
        train_and_save(filepath=args.train)
    elif args.predict:
        result = predict(json.loads(args.predict))
        print(json.dumps(result, indent=2))
    elif args.metrics:
        m = get_metrics(filepath=args.metrics)
        for k, v in m.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"{k:20s}: {v}")
        print("\nConfusion Matrix:\n", np.array(m["confusion_matrix"]))
        print("\nClassification Report:\n", m["classification_report"])
    else:
        parser.print_help()