import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# --- Suppress FutureWarnings from seaborn ---
# Option 1: Suppress all FutureWarnings (cleanest)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Option 2: Or specifically suppress seaborn's deprecation warnings:
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Import feature sets
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

print(f"Using {len(df)} molecules for analysis")

# -------------------- 1. Correlation Matrix --------------------
all_features = list(set(CLASSICAL_FEATURES + BDE_ENHANCED_FEATURES + PHYSICS_INDEPENDENT_FEATURES))
corr_features = [f for f in all_features if f in df.columns and f != 'ZPE_scaling_factor']  # remove constant column

corr_matrix = df[corr_features].corr()

# Print correlations with BDE_min
if 'BDE_min_kJmol' in corr_matrix.columns:
    bde_corr = corr_matrix['BDE_min_kJmol'].sort_values(ascending=False)
    print("\nCorrelation of features with BDE_min_kJmol:")
    print(bde_corr)

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved feature_correlation_matrix.png")

# -------------------- 2. Cross-Validated Predictions --------------------
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

feature_sets = [
    (CLASSICAL_FEATURES, "Classical-Only (A)"),
    (BDE_ENHANCED_FEATURES, "Classical + BDE (A+B)"),
    (PHYSICS_INDEPENDENT_FEATURES, "Classical + Independent Physics (A+C)"),
    (PHYSICS_FULL_FEATURES, "Full (A+B+C)")
]

predictions = {}
errors = {}
kf = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

for features, name in feature_sets:
    X = df[features].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    X = X.fillna(X.median())
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', create_stacked_model())
    ])
    
    y_pred = cross_val_predict(pipeline, X, y, cv=kf)
    predictions[name] = y_pred
    errors[name] = y.values - y_pred

error_df = pd.DataFrame(errors)
error_df['name'] = df['name'].values
error_df['BDE_lit'] = df[TARGET_RAW].values
error_df['BDE_min'] = df['BDE_min_kJmol'].values
error_df['bond_type'] = df['BDE_bond_type_of_min'].values
error_df['oxygen_radical_flag'] = df['oxygen_radical_flag'].values
error_df['has_oxygen'] = df['has_oxygen'].values

error_df.to_csv('model_errors.csv', index=False)
print("Saved model_errors.csv")

# -------------------- 3. Error Analysis Plots (2x2 Layout) --------------------
model_names = [n for _, n in feature_sets]

# 3.1 Boxplots of absolute errors per model
plt.figure(figsize=(10, 6))
abs_errors = error_df[model_names].abs()
sns.boxplot(data=abs_errors)
plt.title('Absolute Prediction Errors by Model (log10 space)')
plt.ylabel('Absolute Error (log10)')
plt.xlabel('Model')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('error_boxplots_by_model.png', dpi=300, bbox_inches='tight')
print("Saved error_boxplots_by_model.png")

# 3.2 Error vs BDE_min (2x2 layout instead of 1x4)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    ax = axes[idx]
    ax.scatter(error_df['BDE_min'], error_df[name], alpha=0.5, s=15)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('BDE_min (kJ/mol)', fontsize=10)
    ax.set_ylabel('Prediction Error (log10)', fontsize=10)
    ax.set_title(name, fontsize=11)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_vs_BDEmin.png', dpi=300, bbox_inches='tight')
print("Saved error_vs_BDEmin.png")

# 3.3 Errors grouped by bond type (2x2 layout)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    ax = axes[idx]
    sns.boxplot(data=error_df, x='bond_type', y=name, hue='bond_type', palette='Set2', 
                legend=False, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bond Type', fontsize=10)
    ax.set_ylabel('Prediction Error (log10)', fontsize=10)
    ax.set_title(name, fontsize=11)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_by_bondtype.png', dpi=300, bbox_inches='tight')
print("Saved error_by_bondtype.png")

# 3.4 Errors grouped by oxygen radical flag (2x2 layout)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    ax = axes[idx]
    sns.boxplot(data=error_df, x='oxygen_radical_flag', y=name, 
                hue='oxygen_radical_flag', palette='Set2', legend=False, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Oxygen Radical Flag', fontsize=10)
    ax.set_ylabel('Prediction Error (log10)', fontsize=10)
    ax.set_title(name, fontsize=11)
    # --- FIX: Set ticks first, then labels ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['False', 'True'])
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_by_oxygenflag.png', dpi=300, bbox_inches='tight')
print("Saved error_by_oxygenflag.png")

# 3.5 Error comparison: A+B vs Full (2x2 layout for better readability)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: A+B vs Full
ax = axes[0]
ax.scatter(error_df["Classical + BDE (A+B)"], error_df["Full (A+B+C)"], 
           alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('Error A+B')
ax.set_ylabel('Error Full (A+B+C)')
ax.set_title('A+B vs Full Model Errors')
ax.legend()
ax.grid(alpha=0.3)

# Right: A vs A+C (to show independent physics hurts)
ax = axes[1]
ax.scatter(error_df["Classical-Only (A)"], error_df["Classical + Independent Physics (A+C)"], 
           alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('Error A')
ax.set_ylabel('Error A+C')
ax.set_title('A vs A+C (Independent Physics)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_comparison_plots.png', dpi=300, bbox_inches='tight')
print("Saved error_comparison_plots.png")

# 3.6 Identify molecules with largest improvement/deterioration
error_df['delta'] = np.abs(error_df["Full (A+B+C)"]) - np.abs(error_df["Classical + BDE (A+B)"])
improved = error_df[error_df['delta'] < -0.01]
worsened = error_df[error_df['delta'] > 0.01]

print(f"\nNumber of molecules improved by physics features (>0.01 log10): {len(improved)}")
print(f"Number of molecules worsened by physics features (>0.01 log10): {len(worsened)}")

if len(improved) > 0:
    print("\nTop improved molecules:")
    print(improved[['name', 'delta', 'BDE_lit', 'BDE_min', 'bond_type', 'oxygen_radical_flag']].sort_values('delta').head(10))

if len(worsened) > 0:
    print("\nTop worsened molecules:")
    print(worsened[['name', 'delta', 'BDE_lit', 'BDE_min', 'bond_type', 'oxygen_radical_flag']].sort_values('delta', ascending=False).head(10))

# 3.7 Summary statistics of errors
print("\nError statistics (absolute error, log10):")
for name in model_names:
    abs_err = np.abs(error_df[name])
    print(f"  {name}: mean={abs_err.mean():.4f}, std={abs_err.std():.4f}, median={abs_err.median():.4f}")

print("\nAnalysis complete. All figures saved.")