import pandas as pd
import numpy as np

# Load all three datasets
df_a = pd.read_csv("group_a_features.csv")
df_b = pd.read_csv("group_b_features.csv")

# --- Load BOTH Group C files ---
df_c_old = pd.read_csv("group_c_features.csv")  # original (dependent on BDE)
df_c_new = pd.read_csv("group_c_features_independent.csv")  # NEW (independent)

# Merge on 'name' (inner join keeps only molecules present in all three)
master = df_a.merge(df_b, on='name').merge(df_c_old, on='name').merge(df_c_new, on='name')

# Add log-transformed physics features from the OLD Group C (if needed for comparison)
EPS = 1e-300
master['log_tunneling_10K'] = np.log10(master['tunneling_p_10K'] + EPS)
master['log_tunneling_20K'] = np.log10(master['tunneling_p_20K'] + EPS)
master['log_tunneling_50K'] = np.log10(master['tunneling_p_50K'] + EPS)
master['log_gamow_factor'] = np.log10(master['gamow_factor'] + EPS)

# Replace infs with NaN
log_cols = ['log_tunneling_10K', 'log_tunneling_20K', 'log_tunneling_50K', 'log_gamow_factor']
for col in log_cols:
    master[col] = master[col].replace([np.inf, -np.inf], np.nan)

# --- Load and merge literature BDEs ---
lit = pd.read_csv("literature_bde.csv")
master = master.merge(lit[['name', 'bde_kj_mol']], on="name", how="left", suffixes=("", "_lit"))
master.rename(columns={"bde_kj_mol": "bde_literature_kjmol"}, inplace=True)

# Save the master dataset
master.to_csv("master_dataset.csv", index=False)
print(f"Master dataset: {len(master)} molecules, {len(master.columns)} columns")
print(f"Literature BDEs available for {master['bde_literature_kjmol'].notna().sum()} molecules")
