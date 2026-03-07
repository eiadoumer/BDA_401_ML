"""
Bank Loan Granting – Streamlit Application
==========================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Loan Granting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load model artefacts (train_model.py must be run first)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model        = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ──────────────────────────────────────────────────────────────────────────────
# Load training data for background visualisation
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "Bank_Loan_Granting.csv"))
    df["CCAvg"] = df["CCAvg"].astype(str).str.replace("/", ".", regex=False).astype(float)
    df.drop(columns=["ID", "ZIP Code"], inplace=True)
    return df

df_full = load_data()

FEATURE_COLS = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – user input
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🧑‍💼 Applicant Information")
st.sidebar.markdown("---")

age        = st.sidebar.slider("Age (years)",            18, 90, 35)
experience = st.sidebar.slider("Work Experience (years)", 0, 43, 10)
income     = st.sidebar.slider("Annual Income ($K)",      8, 224, 60)
family     = st.sidebar.selectbox("Family Size",         [1, 2, 3, 4], index=1)
ccavg      = st.sidebar.slider("Monthly Credit-Card Spend ($K)", 0.0, 10.0, 1.5, step=0.1)
education  = st.sidebar.selectbox(
    "Education Level",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced/Professional"}[x],
)
mortgage   = st.sidebar.slider("Mortgage Value ($K)",  0, 635, 0)

st.sidebar.markdown("**Bank Products Held**")
securities = st.sidebar.checkbox("Securities Account")
cd_account = st.sidebar.checkbox("CD Account")
online     = st.sidebar.checkbox("Online Banking")
creditcard = st.sidebar.checkbox("Bank Credit Card")

predict_btn = st.sidebar.button("🔍  Predict Loan Decision", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Personal Loan Granting System")
st.markdown(
    "This application uses a **XGBoost classifier** trained on 5,000 bank customers "
    "to predict whether a personal loan will be **granted** or **not granted** to an applicant."
)

if not model_loaded:
    st.error(
        "⚠️  Model files not found!  Please run **`python train_model.py`** first "
        "to train and save the model."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    age, experience, income, family, ccavg,
    education, mortgage,
    int(securities), int(cd_account), int(online), int(creditcard)
]], columns=FEATURE_COLS)

if predict_btn:
    prob   = model.predict_proba(input_data)[0][1]
    pred   = int(prob >= 0.5)
    col1, col2, col3 = st.columns(3)

    with col1:
        if pred == 1:
            st.success("### ✅  Loan GRANTED")
        else:
            st.error("### ❌  Loan NOT GRANTED")

    with col2:
        st.metric("Granted Probability", f"{prob*100:.1f}%")

    with col3:
        st.metric("Not Granted Probability", f"{(1-prob)*100:.1f}%")

    # Probability gauge bar
    st.markdown("#### Loan Grant Confidence")
    bar_col, _ = st.columns([3, 1])
    with bar_col:
        fig_gauge, ax_gauge = plt.subplots(figsize=(7, 0.6))
        ax_gauge.barh([0], [prob], color="#27ae60" if pred == 1 else "#e74c3c", height=0.5)
        ax_gauge.barh([0], [1 - prob], left=[prob], color="#ecf0f1", height=0.5)
        ax_gauge.set_xlim(0, 1)
        ax_gauge.axvline(0.5, color="grey", linewidth=1.2, linestyle="--")
        ax_gauge.set_yticks([])
        ax_gauge.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax_gauge.set_xticklabels(["0%", "25%", "50% threshold", "75%", "100%"])
        ax_gauge.set_title("Predicted probability that the loan is granted")
        fig_gauge.tight_layout()
        st.pyplot(fig_gauge)
        plt.close(fig_gauge)

    st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2= st.tabs(["📊 Data Overview", "🌳 Feature Importance"])

# ── Tab 1 – Data Overview ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Summary")
    col_a, col_b = st.columns(2)

    with col_a:
        total      = len(df_full)
        approved   = df_full["Personal Loan"].sum()
        denied     = total - approved
        st.metric("Total Customers", f"{total:,}")
        st.metric("Loans Granted",      f"{approved:,}  ({approved/total*100:.1f}%)")
        st.metric("Loans Not Granted",   f"{denied:,}  ({denied/total*100:.1f}%)")

    with col_b:
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(
            [approved, denied],
            labels=["Granted", "Not Granted"],
            colors=["#27ae60", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax_pie.set_title("Loan Granting Outcome Distribution")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    st.markdown("#### Raw Data Sample")
    st.dataframe(df_full.head(20), use_container_width=True)

    # Income vs CCAvg coloured by target
    st.markdown("#### Income vs Credit-Card Spend (coloured by Loan Status)")
    fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
    denied_df   = df_full[df_full["Personal Loan"] == 0]
    approved_df = df_full[df_full["Personal Loan"] == 1]
    ax_sc.scatter(denied_df["Income"],   denied_df["CCAvg"],   alpha=0.3, s=8,
                  color="#e74c3c", label="Not Granted")
    ax_sc.scatter(approved_df["Income"], approved_df["CCAvg"], alpha=0.6, s=8,
                  color="#27ae60", label="Granted")
    ax_sc.set_xlabel("Annual Income ($K)")
    ax_sc.set_ylabel("Monthly CC Spend ($K)")
    ax_sc.legend()
    ax_sc.set_title("Income vs CCAvg — coloured by Loan Granting Status")
    st.pyplot(fig_sc)
    plt.close(fig_sc)

# ── Tab 2 – Feature Importance ────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest – Feature Importances")
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
        .sort_values("Importance", ascending=True)
    )
    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if v > 0.05 else "#3498db" for v in fi_df["Importance"]]
    ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
    ax_fi.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
    ax_fi.set_title("Feature Importances — Random Forest")
    for i, v in enumerate(fi_df["Importance"]):
        ax_fi.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    fig_fi.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    st.info(
        "**Interpretation:** Features with higher importance contribute more to the "
        "model's splitting decisions.  Income, CCAvg, and CD Account are typically "
        "the strongest predictors of whether a personal loan is granted."
    )


    

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "BDA401 Machine Learning Project | Bank Loan Granting Prediction | "
    "XGBoost Classifier | © 2026"
)
