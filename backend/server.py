from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle 
import os

from eda import datapreparation

app = Flask(__name__)
CORS(app)
DATA_PATH = "customer_churn.csv"
# ── Load models once at startup ──────────────────────────────────
def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
rf_model   = load_pickle("saved_models/random_forest.pkl")
dt_model   = load_pickle("saved_models/decision_tree.pkl")
xgb_model  = load_pickle("saved_models/xgboost_model.pkl")
rf_columns = load_pickle("saved_models/rf_columns.pkl")
dt_columns = load_pickle("saved_models/dt_columns.pkl")
xgb_columns = load_pickle("saved_models/xgb_columns.pkl")

# ── Preprocessing: convert frontend form → model input ──────────
def preprocess_input(form: dict, columns: list) -> pd.DataFrame:
    d = {}

    d["gender"]           = 0 if form.get("gender") == "Male" else 1
    d["SeniorCitizen"]    = int(bool(form.get("seniorCitizen", False)))
    d["Partner"]          = int(bool(form.get("partner", False)))
    d["Dependents"]       = int(bool(form.get("dependents", False)))
    d["PhoneService"]     = int(bool(form.get("phoneService", False)))
    d["MultipleLines"]    = int(bool(form.get("multipleLines", False)))
    d["OnlineSecurity"]   = int(bool(form.get("onlineSecurity", False)))
    d["OnlineBackup"]     = int(bool(form.get("onlineBackup", False)))
    d["DeviceProtection"] = int(bool(form.get("deviceProtection", False)))
    d["TechSupport"]      = int(bool(form.get("techSupport", False)))
    d["StreamingTV"]      = int(bool(form.get("streamingTV", False)))
    d["StreamingMovies"]  = int(bool(form.get("streamingMovies", False)))
    d["PaperlessBilling"] = int(bool(form.get("paperlessBilling", False)))

    tenure          = int(form.get("tenure", 1))
    monthly_charges = float(form.get("monthlyCharges", 0))
    total_charges   = tenure * monthly_charges

    d["MonthlyCharges"] = monthly_charges
    d["TotalCharges"]   = total_charges

    def tenure_group(t):
        if t <= 12:   return 1
        elif t <= 24: return 2
        elif t <= 36: return 3
        elif t <= 48: return 4
        elif t <= 60: return 5
        else:         return 6

    tg = tenure_group(tenure)
    for g in range(1, 7):
        d[f"tenure_group_{g}"] = 1 if tg == g else 0

    for val in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{val}"] = 1 if form.get("internetService") == val else 0

    contract_map = {
        "Month-to-Month": "Month-to-month",
        "One year":        "One year",
        "Two year":        "Two year",
    }
    for val in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{val}"] = 1 if contract_map.get(form.get("contract", "")) == val else 0

    for val in ["Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check"]:
        pm = form.get("paymentMethod", "")
        d[f"PaymentMethod_{val}"] = 1 if pm.lower().replace("(auto)", "(automatic)").lower() == val.lower() else 0

    row = pd.DataFrame([d]).reindex(columns=columns, fill_value=0)
    return row


# ── Survival curve ───────────────────────────────────────────────
def build_survival_curve(churn_prob: float, current_tenure: int) -> dict:
    tenures  = list(range(0, 73, 3))
    lam      = -np.log(max(1 - churn_prob, 1e-6)) / max(current_tenure, 1)
    survival = [float(np.exp(-lam * t)) for t in tenures]
    hazard   = [float(lam * t) for t in tenures]
    return {
        "tenures":        tenures,
        "survival":       survival,
        "hazard":         hazard,
        "current_tenure": current_tenure,
    }


# ── Risk level ───────────────────────────────────────────────────
def risk_level(prob: float) -> str:
    if prob < 0.30:   return "LOW"
    elif prob < 0.50: return "MEDIUM"
    elif prob < 0.75: return "HIGH"
    else:             return "EXTREME"


# ── Feature importance as SHAP-style values ──────────────────────
def get_shap_values(model, row: pd.DataFrame) -> list:
    try:
        importances = None
 
        # Direct model (RF, XGBoost)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
 
        # CalibratedClassifierCV → .estimator
        elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
            importances = model.estimator.feature_importances_
 
        # CalibratedClassifierCV → calibrated_classifiers_[0].estimator (DT fix)
        elif hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "feature_importances_"):
                importances = base.feature_importances_
 
        if importances is None:
            print(f"SHAP: no feature_importances_ found on {type(model)}")
            return []
        features  = row.columns.tolist()
        values    = row.values[0]
        shap_list = [
            {"feature": feat, "shap": round(float(imp * val) if val != 0 else float(imp * 0.01), 4)}
            for feat, imp, val in zip(features, importances, values)
        ]
        shap_list.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return shap_list[:10]
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


# ── Health check ─────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "rf_loaded": rf_model is not None,
        "dt_loaded": dt_model is not None,
        "xgb_loaded": xgb_model is not None,
    })


@app.route("/predict/<model_key>", methods=["POST"])
def predict(model_key):
    try:
        body = request.json

        if model_key == "rf":
            model, columns = rf_model, rf_columns
            model_name = "Random Forest"
        elif model_key == "dt":
            model, columns = dt_model, dt_columns
            model_name = "Decision Tree"
        elif model_key == "xgb":
            model = load_pickle("saved_models/xgboost_model.pkl")
            columns = load_pickle("saved_models/xgb_columns.pkl")
            model_name = "XGBoost"
        else:
            return jsonify({"error": "Invalid model"}), 400

        if model is None or columns is None:
            return jsonify({
                "error": f"Model '{model_key}' not loaded"
            }), 503

        row  = preprocess_input(body, columns)
        pred = int(model.predict(row)[0])
        prob = float(model.predict_proba(row)[0][1])

        tenure          = int(body.get("tenure", 1))
        monthly_charges = float(body.get("monthlyCharges", 0))
        expected_months = max(1, int((1 - prob) * 72))
        ltv             = round(monthly_charges * expected_months)

        return jsonify({
            "prediction": pred,
            "churn_probability": round(prob, 4),
            "label": "Churn" if pred == 1 else "No Churn",
            "risk_level": risk_level(prob),
            "ltv": ltv,
            "model_used": model_name,
            "survival_curve": build_survival_curve(prob, tenure),
            "shap_values": get_shap_values(model, row),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)