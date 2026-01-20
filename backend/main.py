from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# ---------- FastAPI App ----------
app = FastAPI(title="PaySafe UPI Fraud Detection API")

# ---------- Enable CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
MODEL_DIR = "models"
DASHBOARD_DIR = "dashboard"

# ---------- Load Artifacts ----------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))

# ---------- Request Schema ----------
class Transaction(BaseModel):
    transaction_id: str
    amount: float
    upi_id: str
    timestamp: str

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "Welcome to PaySafe Fraud Detection API"}

# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict(transaction: Transaction):
    """
    Accepts transaction details and returns fraud prediction.
    """
    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "transaction_id": transaction.transaction_id,
        "amount": transaction.amount,
        "upi_id": transaction.upi_id,
        "timestamp": transaction.timestamp
    }])

    # --- Preprocessing ---
    # Example: use only numeric features (adjust to your training pipeline)
    numeric_features = ["amount"]
    X = input_data[numeric_features].values

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = xgb_model.predict(X_scaled)[0]
    probability = xgb_model.predict_proba(X_scaled)[0][1]

    return {
        "transaction_id": transaction.transaction_id,
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 3)
    }

# ---------- SHAP Feature Importance Endpoint ----------
@app.get("/shap/importance")
def shap_importance():
    """
    Returns SHAP feature importance from CSV.
    """
    shap_csv = os.path.join(MODEL_DIR, "shap_feature_importance_bp.csv")
    df = pd.read_csv(shap_csv)
    return df.to_dict(orient="records")

# ---------- Dashboard Data Endpoints ----------
@app.get("/dashboard/fraud_vs_nonfraud")
def fraud_vs_nonfraud():
    """
    Returns SHAP fraud vs non-fraud summary.
    """
    csv_path = os.path.join(DASHBOARD_DIR, "shap_fraud_vs_non_fraud.csv")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

@app.get("/dashboard/transaction_values")
def transaction_values():
    """
    Returns SHAP transaction values summary.
    """
    csv_path = os.path.join(DASHBOARD_DIR, "shap_transaction_values.csv")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")