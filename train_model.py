"""
Bank Loan Granting - Model Training Script
==========================================
Trains a Random Forest classifier and saves the model + scaler to disk.
Run this script once before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
)
import joblib
import os

# ── 1. Load & pre‑process ──────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "Bank_Loan_Granting.csv")

df = pd.read_csv(DATA_PATH)

# CCAvg is stored as "1/60" meaning 1.60 — replace "/" with "." and cast to float
df["CCAvg"] = df["CCAvg"].astype(str).str.replace("/", ".", regex=False).astype(float)

# Drop non‑predictive columns
df.drop(columns=["ID", "ZIP Code"], inplace=True)

# Features and target
FEATURE_COLS = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"
]
TARGET_COL = "Personal Loan"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── 2. Train / test split ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 3. Scale features ─────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train models ───────────────────────────────────────────────────────────

models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=None,
                                                  random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Random Forest":
        model.fit(X_train, y_train)           # RF doesn't need scaling
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    results[name] = {"accuracy": acc, "roc_auc": auc, "model": model}
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}   ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Granted", "Granted"]))

# ── 5. Cross‑validation on best model (Random Forest) ─────────────────────────

best_model = models["Random Forest"]
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc")
print(f"\nRandom Forest – 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────

joblib.dump(best_model, "rf_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

print("\n✅  Saved: rf_model.pkl | scaler.pkl | feature_cols.pkl")
print(f"   Feature columns: {FEATURE_COLS}")
