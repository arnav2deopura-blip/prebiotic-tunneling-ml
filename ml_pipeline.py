import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Import ALL feature sets
from feature_sets import (
    CLASSICAL_FEATURES,
    BDE_ENHANCED_FEATURES,
    PHYSICS_INDEPENDENT_FEATURES,
    PHYSICS_FULL_FEATURES
)

# -------------------- Load & Clean Data --------------------
df = pd.read_csv("master_dataset.csv")

# --- Convert categorical columns to numeric ---
bool_cols = ['has_oxygen', 'oxygen_radical_flag']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0)

charge_map = {'positive': 1, 'negative': -1, 'neutral': 0}
if 'molecular_charge' in df.columns:
    df['molecular_charge'] = df['molecular_charge'].map(charge_map).fillna(0)

# --- Target is literature BDE ---
TARGET_RAW = 'bde_literature_kjmol'
df = df.dropna(subset=[TARGET_RAW])
df = df[df[TARGET_RAW] > 0]   # keep only positive BDEs

TARGET_LOG = 'bde_literature_kjmol_log'
df[TARGET_LOG] = np.log10(df[TARGET_RAW])
TARGET = TARGET_LOG

y = df[TARGET]
y = y.replace([np.inf, -np.inf], np.nan).dropna()
df = df.loc[y.index]

print(f"Using {len(df)} molecules with literature BDEs")

# Define the four models to compare
feature_sets = [
    (CLASSICAL_FEATURES, "Classical-Only (A)"),
    (BDE_ENHANCED_FEATURES, "Classical + BDE (A+B)"),
    (PHYSICS_INDEPENDENT_FEATURES, "Classical + Independent Physics (A+C)"),
    (PHYSICS_FULL_FEATURES, "Full (A+B+C)")
]

def create_stacked_model():
    base_learners = [
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    meta_learner = Ridge(alpha=1.0)
    return StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )

def evaluate_model(X, y, model_name):
    # Impute NaNs in features with column median
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    
    if X.isna().any().any():
        print(f"Warning: {model_name} still has NaNs after imputation. Dropping rows with NaN.")
        X = X.dropna()
        y = y.loc[X.index]
    
    kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', create_stacked_model())
    ])
    
    mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    print(f"\n{model_name} Results:")
    print(f"  MAE (mean ± std): {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"  R² (mean ± std): {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    return mae_scores, r2_scores

results = {}
for features, name in feature_sets:
    X = df[features].copy()
    mae, r2 = evaluate_model(X, y, name)
    results[name] = {'MAE': mae, 'R2': r2}

# Compare the full model (A+B+C) against the classical+BDE baseline (A+B)
mae_baseline = results["Classical + BDE (A+B)"]['MAE']
mae_full = results["Full (A+B+C)"]['MAE']
improvement = (mae_baseline.mean() - mae_full.mean()) / mae_baseline.mean() * 100
print(f"\n{'='*50}")
print(f"MAE Improvement with Physics Features (vs A+B): {improvement:.2f}%")
print(f"{'='*50}")

# Optionally, also compare A vs A+C
mae_A = results["Classical-Only (A)"]['MAE']
mae_A_C = results["Classical + Independent Physics (A+C)"]['MAE']
imp_A_C = (mae_A.mean() - mae_A_C.mean()) / mae_A.mean() * 100
print(f"\nMAE Improvement with Independent Physics (vs A only): {imp_A_C:.2f}%")