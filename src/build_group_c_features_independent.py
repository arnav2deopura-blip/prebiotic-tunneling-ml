#!/usr/bin/env python3
"""
Group C: Independent WKB Tunneling Features from Structural Descriptors
Reads group_b_features.csv and literature_bde.csv to calibrate BDE vs bond length.
Computes tunneling probabilities using estimated barrier height from bond length,
bond length as barrier width, and fixed group frequencies.

Outputs group_c_features_independent.csv with features that are independent of BDE_min.
"""

import numpy as np
import pandas as pd
from scipy.constants import hbar, k, atomic_mass
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor

# ---------- Physical Constants ----------
HARTREE_TO_KJMOL = 2625.5
ANGSTROM_TO_M = 1e-10

MASS_H_KG = 1.008 * atomic_mass
MASS_D_KG = 2.014 * atomic_mass

# Group frequencies for X-H bonds (from IR spectroscopy, cm^-1)
FREQ_CM = {
    "C-H": 3000.0,
    "N-H": 3400.0,
    "O-H": 3600.0,
}

# Default linear relationships (fallback if calibration fails)
DEFAULT_REF = {
    "C-H": {"L_ref": 1.09, "BDE_ref": 420.0, "slope": 300.0},   # kJ/mol per Å
    "N-H": {"L_ref": 1.01, "BDE_ref": 460.0, "slope": 250.0},
    "O-H": {"L_ref": 0.96, "BDE_ref": 490.0, "slope": 200.0},
}

# ---------- Helper Functions ----------
def get_particle_mass(name, bond_type):
    if 'D' in name:
        return MASS_D_KG
    return MASS_H_KG

def estimate_bde_from_length(bond_type, bond_length_A, calibrations):
    """
    Use calibrated model if available and physical, else fallback to default.
    """
    if bond_type in calibrations and calibrations[bond_type] is not None:
        intercept, slope = calibrations[bond_type]
        bde_est = intercept + slope * bond_length_A
        # Clamp to physically reasonable range
        return max(100.0, min(700.0, bde_est))
    else:
        # Default: BDE = BDE_ref + slope * (L_ref - L)
        ref = DEFAULT_REF.get(bond_type, {"L_ref": 1.09, "BDE_ref": 420.0, "slope": 300.0})
        L_ref = ref["L_ref"]
        BDE_ref = ref["BDE_ref"]
        slope = ref["slope"]
        bde_est = BDE_ref + slope * (L_ref - bond_length_A)
        return max(100.0, min(700.0, bde_est))

def calibrate_bde_vs_length(df, lit_df):
    """
    Fit a robust linear model for each bond type.
    If the fit is unphysical (slope > 0 for C-H/N-H/O-H, or extreme intercept),
    fall back to the default literature-derived relationship.
    """
    # Merge literature BDEs onto the group_b dataframe
    merged = df.merge(lit_df, on="name", how="inner")
    merged = merged.dropna(subset=["BDE_min_bond_length_A", "bde_kj_mol"])
    
    calibrations = {}
    
    for bond_type in ["C-H", "N-H", "O-H"]:
        sub = merged[merged["BDE_bond_type_of_min"] == bond_type].copy()
        
        # Default values (physically motivated from literature)
        default = DEFAULT_REF.get(bond_type, {"L_ref": 1.09, "BDE_ref": 420.0, "slope": 300.0})
        L_ref = default["L_ref"]
        BDE_ref = default["BDE_ref"]
        slope_default = default["slope"]  # positive means BDE decreases as L increases
        
        # If too few data points, use default
        if len(sub) < 3:
            print(f"Too few data points ({len(sub)}) for {bond_type} – using default.")
            calibrations[bond_type] = None
            continue
        
        X = sub["BDE_min_bond_length_A"].values.reshape(-1, 1)
        y = sub["bde_kj_mol"].values
        
        # Use HuberRegressor (robust to outliers)
        reg = HuberRegressor(epsilon=1.35, max_iter=1000)
        reg.fit(X, y)
        intercept = reg.intercept_
        slope = reg.coef_[0]
        
        # --- Physical sanity checks ---
        # 1. Slope must be negative (shorter bond = stronger)
        #    For C-H, N-H, O-H, this is always true.
        # 2. At L_ref, the predicted BDE must be within [100, 700] kJ/mol
        bde_at_ref = intercept + slope * L_ref
        
        is_physical = (slope < 0) and (100 < bde_at_ref < 700)
        
        if is_physical:
            print(f"Calibrated {bond_type}: BDE = {intercept:.1f} + {slope:.1f} * L (Å)  [robust, physical]")
            calibrations[bond_type] = (intercept, slope)
        else:
            print(f"Calibrated {bond_type} gave unphysical coefficients (slope={slope:.1f}, bde_at_ref={bde_at_ref:.1f}). Using default.")
            # Use default relationship: BDE = BDE_ref + slope_default * (L_ref - L)
            # This is equivalent to intercept = BDE_ref + slope_default * L_ref, slope = -slope_default
            intercept = BDE_ref + slope_default * L_ref
            slope = -slope_default
            calibrations[bond_type] = (intercept, slope)
    
    return calibrations

def compute_barrier_frequency_from_group_freq(bond_type):
    """Return angular frequency omega from group frequency (cm^-1)."""
    freq_cm = FREQ_CM.get(bond_type, 3000.0)
    c = 299792458.0  # m/s
    # Convert cm^-1 to m^-1, then multiply by c to get s^-1, then 2π for angular
    omega = 2.0 * np.pi * c * freq_cm * 100.0  # because 1 m = 100 cm
    return omega

def transmission_parabolic_avg(V0_joules, omega, T):
    if V0_joules <= 0:
        return 1.0
    beta = 1.0 / (k * T)
    def integrand(E):
        exponent = 2.0 * np.pi * (V0_joules - E) / (hbar * omega)
        if exponent > 100:
            T_E = 0.0
        else:
            T_E = 1.0 / (1.0 + np.exp(exponent))
        boltz = np.exp(-beta * E) * beta
        return T_E * boltz
    result, _ = quad(integrand, 0, V0_joules, limit=100)
    return result

def gamow_factor(V0_joules, mass_kg, width_m):
    if V0_joules <= 0:
        return 1.0
    exponent = (2.0 * width_m / hbar) * np.sqrt(2.0 * mass_kg * V0_joules)
    return np.exp(-exponent)

def crossover_temperature(omega):
    return hbar * omega / (2.0 * np.pi * k)

def main():
    print("Loading group_b_features.csv...")
    df_b = pd.read_csv("group_b_features.csv")

    print("Loading literature_bde.csv for calibration...")
    lit = pd.read_csv("literature_bde.csv")

    print("Calibrating BDE vs bond length...")
    calibrations = calibrate_bde_vs_length(df_b, lit)

    rows = []
    for idx, row in df_b.iterrows():
        name = row['name']
        bond_type = row['BDE_bond_type_of_min']
        oxygen_flag = row.get('oxygen_radical_flag', False)
        bond_length_A = row.get('BDE_min_bond_length_A', np.nan)

        # Skip if missing bond length – use default value
        if pd.isna(bond_length_A) or bond_length_A <= 0:
            # Use typical bond length for this bond type
            ref = DEFAULT_REF.get(bond_type, {"L_ref": 1.09})
            bond_length_A = ref["L_ref"]
            print(f"Warning: {name} missing bond length, using default {bond_length_A:.2f} Å")

        # 1. Estimate BDE from bond length
        bde_est = estimate_bde_from_length(bond_type, bond_length_A, calibrations)

        # Corner-cutting correction for oxygen radicals
        if oxygen_flag:
            effective_bde_kj = bde_est / 1.5
        else:
            effective_bde_kj = bde_est

        # 2. Particle mass (H or D)
        mass_kg = get_particle_mass(name, bond_type)
        mass_amu = mass_kg / atomic_mass

        # 3. Barrier width = bond length (in meters)
        width_m = bond_length_A * ANGSTROM_TO_M

        # 4. Convert BDE to J per particle
        V0_joule_per_particle = effective_bde_kj * 1000.0 / 6.02214076e23

        # 5. Barrier frequency from group frequency (independent of BDE)
        omega = compute_barrier_frequency_from_group_freq(bond_type)

        # 6. Tunneling probabilities
        p_10K = transmission_parabolic_avg(V0_joule_per_particle, omega, 10.0)
        p_20K = transmission_parabolic_avg(V0_joule_per_particle, omega, 20.0)
        p_50K = transmission_parabolic_avg(V0_joule_per_particle, omega, 50.0)
        gamow = gamow_factor(V0_joule_per_particle, mass_kg, width_m)
        T_c = crossover_temperature(omega)

        # Log-transform (avoid log(0) with a small epsilon)
        EPS = 1e-300
        rows.append({
            'name': name,
            'crossover_T_independent': T_c,
            'log_tunneling_10K_independent': np.log10(p_10K + EPS),
            'log_tunneling_20K_independent': np.log10(p_20K + EPS),
            'log_tunneling_50K_independent': np.log10(p_50K + EPS),
            'log_gamow_factor_independent': np.log10(gamow + EPS),
            'BDE_est_from_length_kJmol': bde_est,
            'effective_barrier_independent_kJmol': effective_bde_kj,
        })

    df_c = pd.DataFrame(rows)
    df_c.to_csv("group_c_features_independent.csv", index=False)
    print(f"Wrote {len(df_c)} rows to group_c_features_independent.csv")

    # Print summary of calibration coverage
    print("\nCalibration summary:")
    for bt, cal in calibrations.items():
        if cal is None:
            print(f"  {bt}: using default")
        else:
            print(f"  {bt}: intercept={cal[0]:.1f}, slope={cal[1]:.1f}")

if __name__ == "__main__":
    main()
