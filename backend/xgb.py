"""
xgboost_model.py — XGBoost model for Customer Churn prediction
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

from eda import datapreparation

DEFAULT_MODEL_PATH   = "saved_models/xgboost_model.pkl"
DEFAULT_COLUMNS_PATH = "saved_models/xgb_columns.pkl"

def _build_dataset(filepath: str):
    df = datapreparation(filepath)
    X  = df.drop(columns=["Churn"])
    y  = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote_enn = SMOTEENN(random_state=42)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    return X, y, X_train, X_test, y_train, y_test

def train_and_save(
    filepath:     str = "CustomerChurn.csv",
    model_path:   str = DEFAULT_MODEL_PATH,
    columns_path: str = DEFAULT_COLUMNS_PATH,
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    X, y, X_train, X_test, y_train, y_test = _build_dataset(filepath)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 2)
    print(f"Class ratio (neg/pos) = {spw}")

    print("\nFitting baseline XGBoost …")
    xgb_base = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
    xgb_base.fit(X_train, y_train)
    y_pred_base = xgb_base.predict(X_test)
    y_prob_base = xgb_base.predict_proba(X_test)[:, 1]
    print(f"Baseline → Accuracy: {accuracy_score(y_test, y_pred_base):.4f}  Recall: {recall_score(y_test, y_pred_base):.4f}  AUC: {roc_auc_score(y_test, y_prob_base):.4f}")

    print("\nGrid search 1/2 …")
    grid1 = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
        {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 5, 7]},
        cv=3, scoring="recall", n_jobs=-1, verbose=1,
    )
    grid1.fit(X_train, y_train)
    print("Best (stage 1):", grid1.best_params_)
    xgb1    = grid1.best_estimator_
    y_pred1 = xgb1.predict(X_test)
    y_prob1 = xgb1.predict_proba(X_test)[:, 1]
    print(f"Stage1 → Accuracy: {accuracy_score(y_test, y_pred1):.4f}  Recall: {recall_score(y_test, y_pred1):.4f}  AUC: {roc_auc_score(y_test, y_prob1):.4f}")

    print("\nGrid search 2/2 …")
    grid2 = GridSearchCV(
        XGBClassifier(
            n_estimators=grid1.best_params_["n_estimators"],
            learning_rate=grid1.best_params_["learning_rate"],
            max_depth=grid1.best_params_["max_depth"],
            random_state=42, eval_metric="logloss", use_label_encoder=False,
        ),
        {"subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0],
         "reg_alpha": [0, 0.1, 0.5], "reg_lambda": [1, 1.5, 2]},
        cv=3, scoring="recall", n_jobs=-1, verbose=1,
    )
    grid2.fit(X_train, y_train)
    print("Best (stage 2):", grid2.best_params_)
    xgb2    = grid2.best_estimator_
    y_pred2 = xgb2.predict(X_test)
    y_prob2 = xgb2.predict_proba(X_test)[:, 1]
    print(f"Tuned → Accuracy: {accuracy_score(y_test, y_pred2):.4f}  Recall: {recall_score(y_test, y_pred2):.4f}  AUC: {roc_auc_score(y_test, y_prob2):.4f}")

    candidates = {
        "Baseline":     (xgb_base, y_pred_base, y_prob_base),
        "Stage1 Tuned": (xgb1,     y_pred1,     y_prob1),
        "Fully Tuned":  (xgb2,     y_pred2,     y_prob2),
    }
    results = pd.DataFrame([
        {"Model": name, "Recall": round(recall_score(y_test, preds), 4), "AUC-ROC": round(roc_auc_score(y_test, probs), 4)}
        for name, (_, preds, probs) in candidates.items()
    ])
    best_row    = results.sort_values(["Recall", "AUC-ROC"], ascending=False).iloc[0]
    best_name   = best_row["Model"]
    final_model, _, _ = candidates[best_name]
    print(f"\nSelected: {best_name}  Recall={best_row['Recall']}  AUC={best_row['AUC-ROC']}")

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(columns_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Model saved   → {model_path}")
    print(f"Columns saved → {columns_path}")
    return final_model

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(input_data: dict, model_path: str = DEFAULT_MODEL_PATH, columns_path: str = DEFAULT_COLUMNS_PATH) -> dict:
    model = load_model(model_path)
    with open(columns_path, "rb") as f:
        columns = pickle.load(f)
    row  = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    return {"prediction": pred, "probability": round(prob, 4), "label": "Churn" if pred == 1 else "No Churn"}

def get_metrics(filepath: str = "CustomerChurn.csv", model_path: str = DEFAULT_MODEL_PATH) -> dict:
    model = load_model(model_path)
    _, _, _, X_test, _, y_test = _build_dataset(filepath)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost — Customer Churn")
    parser.add_argument("--train",   metavar="CSV")
    parser.add_argument("--predict", metavar="JSON")
    parser.add_argument("--metrics", metavar="CSV")
    args = parser.parse_args()
    if args.train:
        train_and_save(filepath=args.train)
    elif args.predict:
        print(json.dumps(predict(json.loads(args.predict)), indent=2))
    elif args.metrics:
        m = get_metrics(filepath=args.metrics)
        for k, v in m.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"{k}: {v}")
    else:
        parser.print_help()