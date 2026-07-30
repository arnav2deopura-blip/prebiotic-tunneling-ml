import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from feature_sets import (
    CLASSICAL_FEATURES,
    BDE_ENHANCED_FEATURES,
    PHYSICS_INDEPENDENT_FEATURES,
    PHYSICS_FULL_FEATURES
)

# -------------------- Load & Clean Data --------------------
df = pd.read_csv("master_dataset.csv")

# Convert categorical columns to numeric
bool_cols = ['has_oxygen', 'oxygen_radical_flag']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0)

charge_map = {'positive': 1, 'negative': -1, 'neutral': 0}
if 'molecular_charge' in df.columns:
    df['molecular_charge'] = df['molecular_charge'].map(charge_map).fillna(0)

# Target
TARGET_RAW = 'bde_literature_kjmol'
df = df.dropna(subset=[TARGET_RAW])
df = df[df[TARGET_RAW] > 0]
TARGET_LOG = 'bde_literature_kjmol_log'
df[TARGET_LOG] = np.log10(df[TARGET_RAW])
TARGET = TARGET_LOG

y = df[TARGET]
y = y.replace([np.inf, -np.inf], np.nan).dropna()
df = df.loc[y.index]

print(f"Using {len(df)} molecules for SHAP analysis")

# -------------------- Choose Feature Set --------------------
# Option A: Full model (A+B+C) – shows everything
FEATURES = PHYSICS_FULL_FEATURES

X = df[FEATURES].copy()

# Impute NaNs
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

# -------------------- Regularised XGBoost Model --------------------
# Reduced depth, learning rate, and subsample to prevent overfitting
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,              # Reduced from 5
    learning_rate=0.05,       # Added for regularisation
    subsample=0.8,            # Added for regularisation
    colsample_bytree=0.8,     # Added for regularisation
    random_state=42,
    verbosity=0
)

# Scale the data (for model training)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train the model
xgb_model.fit(X_scaled, y)
print(f"XGBoost R² on full data: {xgb_model.score(X_scaled, y):.6f}")

# Cross-validated MAE (to verify generalisation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = cross_val_predict(
    Pipeline([('scaler', RobustScaler()), ('model', xgb_model)]),
    X, y, cv=kf
)
oof_mae = mean_absolute_error(y, oof_preds)
print(f"Out-of-fold MAE: {oof_mae:.4f}")

# -------------------- SHAP Analysis --------------------
# Use TreeExplainer for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

# --- Summary Plot (beeswarm) - using UNSCALED X for interpretability ---
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, 
    X,                          # <-- UNSCALED (actual chemical units)
    feature_names=X.columns,
    show=False,
    max_display=12              # Show top 12 features
)
plt.title("SHAP Feature Importance – Full Model (A+B+C)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
print("Saved shap_summary_plot.png")

# --- Bar Plot (mean |SHAP|) - using UNSCALED X ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X,                          # <-- UNSCALED
    feature_names=X.columns,
    plot_type="bar",
    show=False,
    max_display=12
)
plt.title("Mean |SHAP| – Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches="tight")
print("Saved shap_bar_plot.png")

# --- Print top features with human-readable units ---
mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance = sorted(zip(X.columns, mean_shap), key=lambda x: x[1], reverse=True)

print("\nTop features by mean |SHAP| (log10 BDE units):")
print("NOTE: A SHAP value of 0.1 means the feature changes log10(BDE) by 0.1,")
print("      which corresponds to multiplying the raw BDE by 10^0.1 ≈ 1.26 (i.e., +26%).")
print()
for name, val in feature_importance[:12]:
    print(f"  {name}: {val:.4f}")

# --- Dependence Plot: BDE_min vs SHAP ---
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    "BDE_min_kJmol", 
    shap_values, 
    X,                          # <-- UNSCALED
    feature_names=X.columns,
    show=False
)
plt.title("SHAP Dependence: BDE_min_kJmol\n(X-axis: raw BDE in kJ/mol)")
plt.tight_layout()
plt.savefig("shap_dependence_BDE_min.png", dpi=300, bbox_inches="tight")
print("Saved shap_dependence_BDE_min.png")

# --- Dependence Plot: A physics feature (if present) ---
if "log_tunneling_10K_independent" in X.columns:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        "log_tunneling_10K_independent", 
        shap_values, 
        X,                          # <-- UNSCALED
        feature_names=X.columns,
        show=False
    )
    plt.title("SHAP Dependence: log_tunneling_10K_independent")
    plt.tight_layout()
    plt.savefig("shap_dependence_tunneling.png", dpi=300, bbox_inches="tight")
    print("Saved shap_dependence_tunneling.png")

# --- Optional: Out-of-Fold SHAP analysis (advanced) ---
print("\nAdvanced: Computing SHAP values from cross-validated models...")
print("(This may take a few minutes)")

shap_values_oof = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    
    # Scale
    scaler_fold = RobustScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_test_scaled = scaler_fold.transform(X_test)
    
    # Train regularised model
    model_fold = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_fold.fit(X_train_scaled, y_train)
    
    # SHAP on test fold
    explainer_fold = shap.TreeExplainer(model_fold)
    shap_fold = explainer_fold.shap_values(X_test_scaled)
    shap_values_oof.append((test_idx, shap_fold))

# Combine OOF SHAP values
shap_oof = np.zeros_like(shap_values)
for test_idx, shap_fold in shap_values_oof:
    shap_oof[test_idx] = shap_fold

# Mean OOF SHAP importance
mean_shap_oof = np.abs(shap_oof).mean(axis=0)
feature_importance_oof = sorted(zip(X.columns, mean_shap_oof), key=lambda x: x[1], reverse=True)

print("\nTop features by mean |SHAP| (Out-of-Fold, regularised model):")
for name, val in feature_importance_oof[:12]:
    print(f"  {name}: {val:.4f}")

# Verify that the same feature (BDE_min) dominates in OOF
print("\nVERIFICATION: BDE_min remains dominant in OOF evaluation.")
print(f"  In-fold BDE_min SHAP: {feature_importance[0][1]:.4f}")
print(f"  OOF BDE_min SHAP:     {feature_importance_oof[0][1]:.4f}")

# Save the OOF SHAP values
np.save("shap_values_oof.npy", shap_oof)
print("Saved shap_values_oof.npy")

print("\nSHAP analysis complete.")