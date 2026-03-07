"""
Bank Loan Granting - Model Training Script
==========================================
Trains an XGBoost classifier and saves the model + scaler to disk.
Run this script once before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score
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
    "XGBoost":             XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         use_label_encoder=False, eval_metric="logloss",
                                         random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    if name in ("XGBoost", "Decision Tree"):
        model.fit(X_train, y_train)   # tree-based: scale-invariant
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "f1": f1, "model": model}
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}   F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Granted", "Granted"]))

# ── 5. Cross‑validation on best model (XGBoost) ───────────────────────────────

best_model = models["XGBoost"]
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1")
print(f"\nXGBoost – 5-Fold CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────

joblib.dump(best_model,   "xgb_model.pkl")
joblib.dump(scaler,       "scaler.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

print("\n✅  Saved: xgb_model.pkl | scaler.pkl | feature_cols.pkl")
print(f"   Feature columns: {FEATURE_COLS}")
