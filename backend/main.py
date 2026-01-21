import os
import requests
import pandas as pd
import joblib
from io import StringIO, BytesIO
from fastapi import FastAPI

app = FastAPI(title="PaySafe UPI Fraud Detection API")

# -----------------------------
# Utility functions
# -----------------------------
def load_csv_from_drive(file_id: str) -> pd.DataFrame:
    """Download a CSV from Google Drive using file ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def load_model_from_drive(file_id: str):
    """Download a serialized model (joblib) from Google Drive using file ID."""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# -----------------------------
# Load datasets and models
# -----------------------------
cleaned_data = load_csv_from_drive(os.getenv("CLEANED_DATA_ID"))
metrics_summary = load_csv_from_drive(os.getenv("METRICS_SUMMARY_ID"))

scaler = load_model_from_drive(os.getenv("SCALER_ID"))
isolation_forest = load_model_from_drive(os.getenv("ISOLATION_FOREST_ID"))
xgb_model = load_model_from_drive(os.getenv("XGB_MODEL_ID"))

shap_transaction_values = load_csv_from_drive(os.getenv("SHAP_TRANSACTION_VALUES_ID"))
shap_feature_importance_bp = load_csv_from_drive(os.getenv("SHAP_FEATURE_IMPORTANCE_BP_ID"))
hybrid_eval_test = load_csv_from_drive(os.getenv("HYBRID_EVAL_TEST_ID"))
shap_fraud_vs_non_fraud = load_csv_from_drive(os.getenv("SHAP_FRAUD_VS_NON_FRAUD_ID"))

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "PaySafe UPI Fraud Detection API is running!"}

@app.get("/preview")
def preview(rows: int = 5):
    """Preview first N rows of cleaned dataset."""
    return cleaned_data.head(rows).to_dict(orient="records")

@app.get("/metrics")
def get_metrics():
    """Return metrics summary dataset."""
    return metrics_summary.to_dict(orient="records")

@app.get("/shap/importance")
def shap_importance(rows: int = 10):
    """Return SHAP feature importance values."""
    return shap_feature_importance_bp.head(rows).to_dict(orient="records")

@app.get("/shap/transactions")
def shap_transactions(rows: int = 10):
    """Return SHAP transaction values."""
    return shap_transaction_values.head(rows).to_dict(orient="records")

@app.get("/shap/fraud_vs_nonfraud")
def shap_fraud(rows: int = 10):
    """Return SHAP fraud vs non-fraud breakdown."""
    return shap_fraud_vs_non_fraud.head(rows).to_dict(orient="records")

@app.get("/hybrid_eval")
def hybrid_eval(rows: int = 10):
    """Return hybrid evaluation test results."""
    return hybrid_eval_test.head(rows).to_dict(orient="records")

# -----------------------------
# Example prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(features: dict):
    """
    Run fraud detection prediction using XGBoost model.
    Expects JSON with feature values.
    """
    import numpy as np

    # Convert input dict to DataFrame
    X = pd.DataFrame([features])

    # Scale features
    X_scaled = scaler.transform(X)

    # Isolation Forest anomaly score
    anomaly_score = isolation_forest.decision_function(X_scaled)

    # XGBoost prediction
    prediction = xgb_model.predict(X_scaled)

    return {
        "prediction": int(prediction[0]),
        "anomaly_score": float(anomaly_score[0])
    }
