import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Config and file paths
# -------------------------------
MODELS_DIR = "Models"
DASHBOARDS_DIR = "Dashboard"
DATA_DIR = "Data"

METRICS_PATH = os.path.join(MODELS_DIR, "metrics_summary.csv")
SHAP_FEATURE_PATH = os.path.join(MODELS_DIR, "shap_feature_importance_bp.csv")
FRAUD_VS_NONFRAUD_PATH = os.path.join(DASHBOARDS_DIR, "shap_fraud_vs_nonfraud_bp.csv")
TRANSACTION_VALUES_PATH = os.path.join(DASHBOARDS_DIR, "shap_transaction_values_bp.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_Data.csv")
HYBRID_EVAL_PATH = os.path.join(DATA_DIR, "hybrid_eval_test.csv")


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("PaySafe UPI Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Feature Importance", "Behavior Profiling", "Transaction Drill-Down", "Try your own data"]
)
threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.8, 0.01)

# -------------------------------
# Load resources
# -------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

metrics = load_csv(METRICS_PATH)
shap_feature_df = load_csv(SHAP_FEATURE_PATH)
fraud_vs_nonfraud_df = load_csv(FRAUD_VS_NONFRAUD_PATH)
transaction_values_df = load_csv(TRANSACTION_VALUES_PATH)
cleaned_df = load_csv(CLEANED_DATA_PATH)
hybrid_eval_df = load_csv(HYBRID_EVAL_PATH)


# -------------------------------
# Overview
# -------------------------------
if page == "Overview":
    st.header("📊 Model performance overview")
    cols = st.columns(3)
    cols[0].metric("Accuracy", f"{metrics['Accuracy'][0]:.3f}")
    cols[1].metric("Precision", f"{metrics['Precision'][0]:.3f}")
    cols[2].metric("Recall", f"{metrics['Recall'][0]:.3f}")
    cols = st.columns(3)
    cols[0].metric("F1 Score", f"{metrics['F1'][0]:.3f}")
    cols[1].metric("ROC AUC (Hybrid)", f"{metrics['ROC_AUC_Hybrid'][0]:.3f}")
    cols[2].metric("PR AUC (Hybrid)", f"{metrics['PR_AUC_Hybrid'][0]:.3f}")

    # Symbolic badges
    if metrics['Precision'][0] >= 0.90:
        st.success("✅ Precision strong")
    if metrics['Recall'][0] < 0.70:
        st.warning("⚠️ Recall could be improved")
    if metrics['F1'][0] < 0.60:
        st.error("🚨 F1 score weak")

# -------------------------------
# Feature importance
# -------------------------------
elif page == "Feature Importance":
    st.header("🔍 Global feature importance (SHAP)")
    st.bar_chart(importance.set_index("Feature"))
    st.caption("Higher mean absolute SHAP indicates stronger influence on predictions.")

# -------------------------------
# Behavior profiling
# -------------------------------
elif page == "Behavior Profiling":
    st.header("🧠 Behavior profiling signals")
    if "BehaviorRisk" in transactions.columns:
        st.write("Behavior risk distribution")
        st.line_chart(transactions["BehaviorRisk"].value_counts().sort_index())  # simple view
    else:
        st.info("BehaviorRisk not found in transactions CSV. Ensure Step 3.5 outputs include it.")

    st.subheader("Fraud vs Non-Fraud SHAP averages")
    st.dataframe(comparison)

# -------------------------------
# Transaction drill-down
# -------------------------------
elif page == "Transaction Drill-Down":
    st.header("🚨 Transaction drill-down with alerts")
    # Apply dynamic threshold on-the-fly
    if "HybridScore" in transactions.columns:
        tx = transactions.copy()
        tx["AlertDynamic"] = (tx["HybridScore"] >= threshold).astype(int)
        st.write(f"Threshold: {threshold:.2f} | Alerts: {tx['AlertDynamic'].sum()} / {len(tx)}")
        st.dataframe(tx[["HybridScore", "IsFraud", "Alert", "AlertDynamic"]].head(30))
        idx = st.number_input("Row index to inspect", 0, len(tx)-1, 0)
        st.write("Selected transaction:", tx.iloc[int(idx)])
    else:
        st.error("HybridScore not found in transactions CSV.")

# -------------------------------
# Try your own data (live scoring)
# -------------------------------
elif page == "Try your own data":
    st.header("🧪 Try your own transaction(s)")
    st.write("Upload CSV with the model's expected feature columns or fill the form below.")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        user_df = pd.read_csv(uploaded)
        st.write("Preview:", user_df.head())

        # Load models
        xgb_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
        iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")

        if not (os.path.exists(xgb_path) and os.path.exists(iso_path)):
            st.error("Model files not found. Ensure xgb_model.joblib and isolation_forest.joblib exist in models/")
        else:
            xgb = joblib.load(xgb_path)
            iso = joblib.load(iso_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

            # Assume user_df already has engineered columns matching training features
            feature_cols = importance["Feature"].tolist()  # robust: use SHAP features as reference
            missing = [c for c in feature_cols if c not in user_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X_user = user_df[feature_cols].copy()

                # Optional scaling if used in training for numeric cols only
                if scaler is not None:
                    # Select numeric columns present in scaler’s training
                    # Here, we simply pass numeric types; adapt if you saved exact column list
                    num_cols = X_user.select_dtypes(include=[np.number]).columns
                    X_user[num_cols] = scaler.transform(X_user[num_cols])

                # Scores
                xgb_prob = xgb.predict_proba(X_user)[:, 1]
                iso_score_raw = -iso.decision_function(X_user)
                iso_min, iso_max = iso_score_raw.min(), iso_score_raw.max()
                iso_score = (iso_score_raw - iso_min) / (iso_max - iso_min + 1e-9)

                # Rule boost if present
                rb = 0
                if "RuleHighValue" in X_user.columns and "RuleRapidFire" in X_user.columns:
                    rb = 0.5*X_user["RuleHighValue"].values + 0.5*X_user["RuleRapidFire"].values
                    rb = np.clip(rb, 0, 1)

                hybrid = 0.6*xgb_prob + 0.3*iso_score + 0.1*rb
                alert = (hybrid >= threshold).astype(int)

                out = user_df.copy()
                out["HybridScore"] = hybrid
                out["Alert"] = alert
                st.success("✅ Scored successfully")
                st.dataframe(out.head(20))
                st.download_button("Download scored CSV", out.to_csv(index=False), "scored_transactions.csv", "text/csv")