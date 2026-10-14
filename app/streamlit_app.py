"""
streamlit_app.py
----------------
Customer Churn Prediction Web App

A beautiful, interactive web app where you can:
  1. Enter a customer's details in the sidebar form
  2. Click "Predict" to see their churn risk
  3. View the churn probability, risk level, and top reasons
  4. Get recommended retention actions

Run with:
    streamlit run app/streamlit_app.py

Author: Churn Prediction Project
"""

import sys
from pathlib import Path

# ── Make src importable when running from the app/ directory ───────────────
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from src.predict import predict_churn, load_model, get_risk_level
from src.features import engineer_features


# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover { transform: translateY(-3px); }

    .risk-high {
        background: linear-gradient(135deg, rgba(233,30,99,0.3), rgba(233,30,99,0.1));
        border: 1px solid rgba(233,30,99,0.5);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(255,152,0,0.3), rgba(255,152,0,0.1));
        border: 1px solid rgba(255,152,0,0.5);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(76,175,80,0.3), rgba(76,175,80,0.1));
        border: 1px solid rgba(76,175,80,0.5);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .insight-box {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #7C4DFF;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    .action-box {
        background: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
    }

    h1 { color: white !important; }
    h2, h3 { color: rgba(255,255,255,0.9) !important; }
    p, li { color: rgba(255,255,255,0.8) !important; }

    .stButton > button {
        background: linear-gradient(90deg, #7C4DFF, #E040FB);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: opacity 0.2s ease;
    }

    .stButton > button:hover { opacity: 0.85; }

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #FF9800, #E91E63);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load SHAP explainer (cached for performance) ──────────────────
@st.cache_resource(show_spinner=False)
def load_cached_model():
    """Load the model once and cache it across sessions."""
    model_path = ROOT_DIR / "models" / "xgboost_pipeline.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_shap_values(model, customer_df: pd.DataFrame):
    """
    Compute SHAP values for a single customer to explain the prediction.

    SHAP tells us: "which features pushed the prediction UP (toward churn)
    and which pushed it DOWN (toward staying)?"
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        X_transformed = preprocessor.transform(customer_df)

        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_transformed)

        # Get feature names
        num_features = preprocessor.transformers_[0][2]
        ohe = preprocessor.named_transformers_["cat"]["onehot"]
        cat_features = ohe.get_feature_names_out().tolist()
        all_features = num_features + cat_features

        # Create a Series of shap_value → feature_name
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # For binary classification

        shap_series = pd.Series(shap_vals[0], index=all_features)
        return shap_series
    except Exception as e:
        return None


def plot_shap_bar(shap_series: pd.Series, top_n: int = 8):
    """Plot a horizontal bar chart of top SHAP values."""
    top = shap_series.abs().nlargest(top_n)
    top_vals = shap_series[top.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E91E63" if v > 0 else "#4CAF50" for v in top_vals.values]
    top_vals.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.set_xlabel("SHAP Value (impact on churn probability)", color="white", fontsize=10)
    ax.set_title("Top Factors Driving This Prediction", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color((1.0, 1.0, 1.0, 0.2))   # matplotlib needs tuple, not CSS rgba()
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)

    for i, (val, patch) in enumerate(zip(top_vals.sort_values().values, ax.patches)):
        label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        ax.text(val + (0.002 if val >= 0 else -0.002), patch.get_y() + patch.get_height() / 2,
                label, va="center", ha="left" if val >= 0 else "right",
                color="white", fontsize=9)

    plt.tight_layout()
    return fig


def format_feature_name(name: str) -> str:
    """Convert OHE feature names like 'Contract_Month-to-month' to readable text."""
    if "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]} = {parts[1]}"
    return name


def get_retention_actions(shap_series: pd.Series, customer: dict) -> list:
    """Generate targeted retention recommendations based on top churn drivers."""
    actions = []
    if customer.get("Contract") == "Month-to-month":
        actions.append("📋 **Offer a discounted 1-year or 2-year contract** — long-term contracts dramatically reduce churn risk.")
    if customer.get("InternetService") == "Fiber optic" and customer.get("TechSupport") == "No":
        actions.append("🛠️ **Offer a free 3-month TechSupport trial** — Fiber customers without tech support churn at 2× the rate.")
    if customer.get("OnlineSecurity") == "No":
        actions.append("🔒 **Bundle Online Security at a discount** — security add-ons increase perceived value and stickiness.")
    if customer.get("tenure", 100) < 12:
        actions.append("🎁 **Trigger a loyalty milestone reward** — customers in their first year need engagement. Offer a loyalty bonus or service upgrade.")
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append("💳 **Incentivize auto-pay enrollment** — customers on electronic check have higher churn. Offer a $5/month discount for credit card / bank transfer auto-pay.")
    if len(actions) == 0:
        actions.append("📞 **Proactive check-in call** — assign an account manager to reach out and ensure satisfaction.")
        actions.append("🎯 **Personalized loyalty offer** — tailor a discount or upgrade based on their current plan.")
    return actions[:3]  # Top 3 actions


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 10px 0;">
    <h1 style="font-size:2.8em; font-weight:700; 
               background: linear-gradient(90deg, #7C4DFF, #E040FB, #FF4081);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📡 Customer Churn Predictor
    </h1>
    <p style="font-size:1.1em; color:rgba(255,255,255,0.6); margin-top:-10px;">
        Powered by XGBoost · IBM Telco Dataset · SHAP Explainability
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Check model is available ───────────────────────────────────────────────
model = load_cached_model()

if model is None:
    st.error("""
    ⚠️ **Model not found!**

    Please train the model first by running:
    ```bash
    python -m src.train
    ```
    Then restart this app.
    """)
    st.stop()

# ── Sidebar: Customer Input Form ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Customer Profile")
    st.markdown("Fill in the customer details below:")

    st.markdown("### 📋 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    partner = st.selectbox("Has Partner?", ["Yes", "No"], key="partner")
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"], key="dependents")

    st.markdown("### 📅 Account")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, key="tenure")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
    paperless = st.selectbox("Paperless Billing?", ["Yes", "No"], key="paperless")
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        key="payment",
    )

    st.markdown("### 📞 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
    multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multi")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="internet")

    if internet != "No":
        st.markdown("**Add-on Services:**")
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="sec")
        online_bk = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="bk")
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="dev")
        tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="mov")
    else:
        online_sec = online_bk = device_prot = tech_sup = streaming_tv = streaming_movies = "No internet service"

    st.markdown("### 💰 Charges")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5, key="monthly")

    # Auto-calculate total charges as a reasonable estimate
    estimated_total = round(monthly_charges * max(tenure, 1) * 0.95, 2)
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(estimated_total),
        step=1.0,
        key="total",
        help="Auto-estimated as ~95% × Monthly × Tenure. Adjust if known.",
    )

    st.divider()
    predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True)


# ── Main Panel: Results ────────────────────────────────────────────────────
if not predict_button:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>Accurate Predictions</h3>
            <p>XGBoost model trained on 7,043 telecom customers with ~83% ROC-AUC</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🔍</h2>
            <h3>Explainable AI</h3>
            <p>SHAP values reveal exactly WHY a customer is predicted to churn</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>💡</h2>
            <h3>Action-Ready</h3>
            <p>Personalized retention recommendations for each at-risk customer</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="text-align:center; color:rgba(255,255,255,0.5); font-size:1em;">
        👈 Fill in the customer form in the sidebar and click <strong>Predict Churn Risk</strong>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Validate inputs ────────────────────────────────────────────────
    errors = []
    if monthly_charges <= 0:
        errors.append("Monthly Charges must be greater than 0.")
    if total_charges < 0:
        errors.append("Total Charges cannot be negative.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
        st.stop()

    # ── Build customer dict ────────────────────────────────────────────
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bk,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # ── Run prediction ─────────────────────────────────────────────────
    with st.spinner("🤖 Analyzing customer profile..."):
        try:
            result = predict_churn(customer, model_path=ROOT_DIR / "models" / "xgboost_pipeline.pkl")
            prob = result["churn_probability"]
            risk = result["risk_level"]
            will_churn = result["will_churn"]

            # Compute SHAP values for explanation
            customer_df = pd.DataFrame([customer])
            customer_df = engineer_features(customer_df)
            shap_series = get_shap_values(model, customer_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
            st.stop()

    # ── Results Layout ─────────────────────────────────────────────────
    risk_colors = {"High": "#E91E63", "Medium": "#FF9800", "Low": "#4CAF50"}
    risk_emojis = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    risk_css   = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}

    col_prob, col_risk, col_verdict = st.columns(3)

    with col_prob:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="font-size:2.5em; color:{risk_colors[risk]}; font-weight:700;">
                {prob*100:.1f}%
            </h2>
            <p style="font-size:1em; color:rgba(255,255,255,0.7);">Churn Probability</p>
        </div>""", unsafe_allow_html=True)

    with col_risk:
        st.markdown(f"""
        <div class="{risk_css[risk]}">
            <h2 style="font-size:2.5em; margin:0;">{risk_emojis[risk]}</h2>
            <h3 style="color:{risk_colors[risk]}; margin:5px 0;">{risk} Risk</h3>
            <p style="color:rgba(255,255,255,0.7); margin:0;">Churn Category</p>
        </div>""", unsafe_allow_html=True)

    with col_verdict:
        verdict = "⚠️ Will Churn" if will_churn else "✅ Will Stay"
        verdict_color = risk_colors[risk] if will_churn else "#4CAF50"
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="font-size:1.8em; color:{verdict_color}; margin:0;">{verdict}</h2>
            <p style="color:rgba(255,255,255,0.7);">Prediction</p>
        </div>""", unsafe_allow_html=True)

    # ── Probability gauge (progress bar) ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Churn Probability: {prob*100:.1f}%**")
    st.progress(prob)

    st.markdown("""
    <p style="font-size:0.85em; color:rgba(255,255,255,0.4);">
        🟢 0–35% Low risk &nbsp;&nbsp;|&nbsp;&nbsp; 🟡 35–60% Medium risk &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 60–100% High risk
    </p>""", unsafe_allow_html=True)

    st.divider()

    # ── SHAP Explanation ───────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🔍 Why This Prediction?")
        if shap_series is not None:
            fig = plot_shap_bar(shap_series, top_n=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Top 3 drivers as text
            st.markdown("**Top churn drivers:**")
            top3 = shap_series.abs().nlargest(3).index
            for feat in top3:
                val = shap_series[feat]
                direction = "↑ increases" if val > 0 else "↓ decreases"
                display = format_feature_name(feat)
                st.markdown(f"""
                <div class="insight-box">
                    {'🔴' if val > 0 else '🟢'} <strong>{display}</strong>
                    — {direction} churn risk by <strong>{abs(val):.3f}</strong>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("SHAP explanation unavailable for this prediction.")

    with col_right:
        # ── Retention Actions ──────────────────────────────────────────
        st.markdown("### 💡 Recommended Actions")
        actions = get_retention_actions(shap_series if shap_series is not None else pd.Series(), customer)
        for i, action in enumerate(actions, 1):
            st.markdown(f"""
            <div class="action-box">
                <strong>Action {i}:</strong><br>
                {action}
            </div>""", unsafe_allow_html=True)

        # ── Customer Summary ───────────────────────────────────────────
        st.markdown("### 📊 Profile Summary")
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | Tenure | {tenure} months |
        | Contract | {contract} |
        | Monthly Charges | ${monthly_charges:.2f} |
        | Internet | {internet} |
        | Payment | {payment} |
        """)

    st.divider()
    st.markdown("""
    <p style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8em;">
        Built with ❤️ using XGBoost + SHAP + Streamlit · IBM Telco Customer Churn Dataset
    </p>""", unsafe_allow_html=True)
