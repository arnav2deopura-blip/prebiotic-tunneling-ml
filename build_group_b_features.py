import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit terminal noise
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_group_b_features")

HARTREE_TO_KJMOL = 2625.5
ZPE_SCALING_FACTOR = 1.0
GFN2_XTB_H_ATOM_ENERGY = -0.393482811632  # Hartree

REACTIVE_BOND_SMARTS = {
    "C-H": "[#6]-[#1]",
    "N-H": "[#7]-[#1]",
    "O-H": "[#8]-[#1]",
}

# Spin-tax correction coefficients for GFN2-xTB BDE calculations (kJ/mol)
SPIN_TAX_COEFFICIENTS = {
    "intercept": 0.0,
    "slope": -15.4  # Empirical spin correction shift in kJ/mol per unpaired electron created
}

def spin_tax_correction(delta_unpaired_electrons: int) -> float:
    if SPIN_TAX_COEFFICIENTS is None:
        raise NotImplementedError("SPIN_TAX_COEFFICIENTS must be defined.")
    return SPIN_TAX_COEFFICIENTS["intercept"] + (SPIN_TAX_COEFFICIENTS["slope"] * delta_unpaired_electrons)

# Regex to extract total energy from xTB output
ENERGY_RE = re.compile(r"TOTAL ENERGY\s+[:|]?\s*(-?\d+\.\d+)", re.IGNORECASE)

_H_ATOM_ENERGY_CACHE = {}


def get_total_valence_electrons(mol, net_formal_charge=0):
    total_z = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
    return total_z - net_formal_charge


def get_species_charge_and_uhf(mol, net_formal_charge=0):
    ne = get_total_valence_electrons(mol, net_formal_charge)
    charge = int(net_formal_charge)
    uhf = 0 if ne % 2 == 0 else 1
    return charge, uhf


def load_inputs(candidates_path, lit_ref_path):
    candidates = pd.read_csv(candidates_path)
    if os.path.exists(lit_ref_path):
        lit_ref = pd.read_csv(lit_ref_path)
    else:
        lit_ref = pd.DataFrame(columns=["name", "bond_type", "bde_kj_mol", "citation_key", "notes"])
    return candidates, lit_ref


def enumerate_xh_bonds(mol):
    mol_h = Chem.AddHs(mol)
    bonds = []
    for bond_type, smarts in REACTIVE_BOND_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        for match in mol_h.GetSubstructMatches(patt):
            heavy_idx, h_idx = match
            bonds.append({"bond_type": bond_type, "heavy_atom_idx": heavy_idx, "h_atom_idx": h_idx})
    return mol_h, bonds


def lookup_literature(name, bond_type, lit_ref):
    if lit_ref.empty:
        return None
    matches = lit_ref[(lit_ref["name"] == name) & (lit_ref["bond_type"] == bond_type)]
    if len(matches) == 0:
        return None
    return float(matches.iloc[0]["bde_kj_mol"])


def embed_and_optimize(mol, random_seed=0xF00D):
    mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = random_seed

    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=random_seed, useRandomCoords=True) != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return mol


def build_radical_fragment(mol_h, heavy_atom_idx, h_atom_idx):
    rw = Chem.RWMol(mol_h)
    rw.RemoveAtom(h_atom_idx)
    new_heavy_idx = heavy_atom_idx if heavy_atom_idx < h_atom_idx else heavy_atom_idx - 1
    atom = rw.GetAtomWithIdx(new_heavy_idx)
    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
    if atom.GetNumExplicitHs() > 0:
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        pass
    return frag, new_heavy_idx


def mol_to_xyz_block(mol, comment=""):
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)


def run_xtb_energy(xyz_block, charge, uhf, workdir, opt=True):
    xyz_path = Path(workdir) / "mol.xyz"
    xyz_path.write_text(xyz_block)
    xtb_path = shutil.which("xtb")
    if xtb_path is None:
        return None, "xtb executable not found on PATH (checked with shutil.which)"
    cmd = [xtb_path, str(xyz_path), "--chrg", str(charge), "--uhf", str(uhf)]
    if opt:
        cmd.append("--opt")
    try:
        result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        return None, "xtb executable not found on PATH"

    if result.returncode != 0:
        err_msg = result.stderr.strip().split("\n")[-1] if result.stderr else f"exit code {result.returncode}"
        return None, f"xTB failed: {err_msg}"

    matches = ENERGY_RE.findall(result.stdout)
    if not matches:
        return None, "could not parse total energy from xTB output"
    # Take the LAST occurrence (final converged energy)
    return float(matches[-1]), None


def get_h_atom_energy(workdir):
    if "energy" in _H_ATOM_ENERGY_CACHE and _H_ATOM_ENERGY_CACHE["energy"] is not None:
        return _H_ATOM_ENERGY_CACHE["energy"], None

    xyz_block = "1\nH atom\nH 0.000000 0.000000 0.000000"
    energy, err = run_xtb_energy(xyz_block, charge=0, uhf=1, workdir=workdir, opt=False)

    if energy is None:
        logger.warning(f"xTB single-point energy on H atom failed ({err}). Using standard GFN2-xTB reference ({GFN2_XTB_H_ATOM_ENERGY} Eh).")
        energy = GFN2_XTB_H_ATOM_ENERGY

    _H_ATOM_ENERGY_CACHE["energy"] = energy
    return energy, None


def compute_bde_xtb(mol, bond, net_formal_charge, workdir):
    heavy_idx, h_idx = bond["heavy_atom_idx"], bond["h_atom_idx"]

    mol_h = Chem.AddHs(mol)
    parent_3d = embed_and_optimize(mol_h)
    if parent_3d is None:
        return {"bde_kj_mol": None, "flag": "embed_failed", "detail": "parent embed failed"}

    # --- NEW: Extract bond length from the 3D conformation ---
    bond_length_angstrom = np.nan
    try:
        conf = parent_3d.GetConformer()
        heavy_pos = conf.GetAtomPosition(heavy_idx)
        h_pos = conf.GetAtomPosition(h_idx)
        bond_length_angstrom = heavy_pos.Distance(h_pos)  # in angstroms
    except Exception:
        pass  # Keep as nan if anything fails

    frag, new_heavy_idx = build_radical_fragment(mol_h, heavy_idx, h_idx)
    frag_3d = embed_and_optimize(frag)
    if frag_3d is None:
        return {"bde_kj_mol": None, "flag": "embed_failed", "detail": "radical fragment embed failed"}

    # Determine charge and UHF for parent and fragment
    parent_charge, parent_uhf = get_species_charge_and_uhf(parent_3d, net_formal_charge)
    frag_charge, frag_uhf = get_species_charge_and_uhf(frag_3d, net_formal_charge)

    # Single‑atom fragment: skip geometry optimization
    opt_frag = False if frag_3d.GetNumAtoms() == 1 else True

    # Get H atom energy (cached)
    h_energy, h_err = get_h_atom_energy(workdir)

    # Run xtb on parent (always optimize)
    parent_energy, parent_err = run_xtb_energy(
        mol_to_xyz_block(parent_3d, "parent"),
        parent_charge, parent_uhf, workdir, opt=True
    )
    if parent_energy is None:
        return {"bde_kj_mol": None, "flag": "xtb_energy_missing", "detail": f"Parent xTB failed: {parent_err}"}

    # Run xtb on fragment (use opt_frag)
    frag_energy, frag_err = run_xtb_energy(
        mol_to_xyz_block(frag_3d, "radical_fragment"),
        frag_charge, frag_uhf, workdir, opt=opt_frag
    )
    if frag_energy is None:
        return {"bde_kj_mol": None, "flag": "xtb_energy_missing", "detail": f"Radical xTB failed: {frag_err}"}

    bde_hartree = (frag_energy + h_energy) - parent_energy
    return {
        "bde_kj_mol": bde_hartree * HARTREE_TO_KJMOL,
        "flag": None,
        "radical_mol": frag,
        "radical_atom_idx": new_heavy_idx,
        "bond_length_angstrom": bond_length_angstrom,   # NEW: structural bond length
    }


def is_oxygen_radical(mol_h, radical_atom_idx):
    atom = mol_h.GetAtomWithIdx(radical_atom_idx)
    if atom.GetSymbol() == "O":
        return True
    return any(n.GetSymbol() == "O" for n in atom.GetNeighbors())


def radical_stability_score(mol_h, radical_atom_idx):
    atom = mol_h.GetAtomWithIdx(radical_atom_idx)
    score = 0.0

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) or other.GetIsAromatic():
            score += 2.0
            break

    heteroatom_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() in ("N", "O", "S"))
    score += min(heteroatom_neighbors, 2) * 1.0

    heavy_neighbor_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != "H")
    score += max(heavy_neighbor_count - 1, 0) * 1.0

    return score


def process_bond(name, mol, mol_h, bond, net_formal_charge, lit_ref, workdir, review_log):
    lit_bde = lookup_literature(name, bond["bond_type"], lit_ref)
    if lit_bde is not None:
        radical_atom_idx = bond["heavy_atom_idx"]
        return {
            "bond_type": bond["bond_type"],
            "bde_kj_mol": lit_bde,
            "source": "literature",
            "spin_tax_correction_kj_mol": 0.0,
            "oxygen_radical_flag": is_oxygen_radical(mol_h, radical_atom_idx),
            "radical_stability_score": radical_stability_score(mol_h, radical_atom_idx),
            "bond_length_angstrom": np.nan,   # no 3D structure for literature values
        }

    xtb_result = compute_bde_xtb(mol, bond, net_formal_charge, workdir)
    if xtb_result.get("bde_kj_mol") is None:
        review_log.append({
            "name": name,
            "bond_type": bond["bond_type"],
            "heavy_atom_idx": bond["heavy_atom_idx"],
            "reason": xtb_result.get("flag"),
            "detail": xtb_result.get("detail"),
        })
        return None

    radical_mol = xtb_result["radical_mol"]
    radical_atom_idx = xtb_result["radical_atom_idx"]
    oxy_flag = is_oxygen_radical(radical_mol, radical_atom_idx)

    try:
        correction = spin_tax_correction(delta_unpaired_electrons=1)
    except NotImplementedError:
        correction = 0.0

    bde = xtb_result["bde_kj_mol"]
    if correction is not None:
        bde += correction

    return {
        "bond_type": bond["bond_type"],
        "bde_kj_mol": bde,
        "source": "xtb_corrected",
        "spin_tax_correction_kj_mol": correction,
        "oxygen_radical_flag": oxy_flag,
        "radical_stability_score": radical_stability_score(radical_mol, radical_atom_idx),
        "bond_length_angstrom": xtb_result.get("bond_length_angstrom", np.nan),   # NEW
    }


def process_molecule(row, lit_ref, workdir, review_log):
    name = row["name"]
    smiles = row["smiles"]

    net_formal_charge = 0
    for col in ["net_formal_charge", "formal_charge", "charge", "net_charge"]:
        if col in row and pd.notna(row[col]):
            try:
                net_formal_charge = int(row[col])
                break
            except (ValueError, TypeError):
                pass

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES for {name}: {smiles}")
        return None

    mol_h, bonds = enumerate_xh_bonds(mol)
    if not bonds:
        return {
            "name": name,
            "BDE_min_kJmol": None,
            "BDE_mean_kJmol": None,
            "BDE_bond_type_of_min": None,
            "BDE_source": None,
            "spin_tax_correction_kJmol": None,
            "oxygen_radical_flag": None,
            "ZPE_scaling_factor": ZPE_SCALING_FACTOR,
            "radical_stability_score": None,
            "BDE_min_bond_length_A": None,   # NEW
            "_bond_bdes": [],
        }

    results = []
    for bond in bonds:
        r = process_bond(name, mol, mol_h, bond, net_formal_charge, lit_ref, workdir, review_log)
        if r is not None:
            results.append(r)

    if not results:
        return {
            "name": name,
            "BDE_min_kJmol": None,
            "BDE_mean_kJmol": None,
            "BDE_bond_type_of_min": None,
            "BDE_source": None,
            "spin_tax_correction_kJmol": None,
            "oxygen_radical_flag": None,
            "ZPE_scaling_factor": ZPE_SCALING_FACTOR,
            "radical_stability_score": None,
            "BDE_min_bond_length_A": None,   # NEW
            "_bond_bdes": [],
        }

    min_r = min(results, key=lambda r: r["bde_kj_mol"])
    return {
        "name": name,
        "BDE_min_kJmol": min_r["bde_kj_mol"],
        "BDE_mean_kJmol": float(np.mean([r["bde_kj_mol"] for r in results])),
        "BDE_bond_type_of_min": min_r["bond_type"],
        "BDE_source": min_r["source"],
        "spin_tax_correction_kJmol": min_r["spin_tax_correction_kj_mol"],
        "oxygen_radical_flag": min_r["oxygen_radical_flag"],
        "ZPE_scaling_factor": ZPE_SCALING_FACTOR,
        "radical_stability_score": min_r["radical_stability_score"],
        "BDE_min_bond_length_A": min_r.get("bond_length_angstrom", np.nan),   # NEW
        "_bond_bdes": [r["bde_kj_mol"] for r in results],
    }


def apply_weak_bond_cutoff(df, quartile):
    if df.empty or "BDE_min_kJmol" not in df.columns:
        df["BDE_weak_bond_count"] = 0
        return df, 0.0

    valid_bdes = df["BDE_min_kJmol"].dropna()
    if valid_bdes.empty:
        df["BDE_weak_bond_count"] = 0
        return df, 0.0

    cutoff = valid_bdes.quantile(quartile)
    df["BDE_weak_bond_count"] = df["_bond_bdes"].apply(
        lambda bdes: sum(1 for b in bdes if b is not None and b < cutoff)
    )
    return df, cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="group_a_features.csv")
    parser.add_argument("--lit-ref", default="bde_literature_reference.csv")
    parser.add_argument("--out", default="group_b_features.csv")
    parser.add_argument("--review-out", default="group_b_manual_review.csv")
    parser.add_argument("--weak-bond-quartile", type=float, default=0.25)
    args = parser.parse_args()

    candidates, lit_ref = load_inputs(args.candidates, args.lit_ref)

    rows = []
    review_log = []
    with tempfile.TemporaryDirectory() as workdir:
        for idx, row in candidates.iterrows():
            result = process_molecule(row, lit_ref, workdir, review_log)
            if result is not None:
                rows.append(result)
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(candidates)} candidate molecules...")

    out_df = pd.DataFrame(rows)
    out_df, cutoff = apply_weak_bond_cutoff(out_df, args.weak_bond_quartile)
    logger.info(f"Weak-bond BDE cutoff (bottom {args.weak_bond_quartile:.0%} of BDE_min_kJmol): {cutoff:.2f} kJ/mol")

    cols_to_keep = [
        "name", "BDE_min_kJmol", "BDE_mean_kJmol", "BDE_bond_type_of_min",
        "BDE_weak_bond_count", "BDE_source", "spin_tax_correction_kJmol",
        "oxygen_radical_flag", "ZPE_scaling_factor", "radical_stability_score",
        "BDE_min_bond_length_A",   # NEW
    ]
    for col in cols_to_keep:
        if col not in out_df.columns:
            out_df[col] = None

    out_df = out_df[cols_to_keep]
    out_df.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(out_df)} rows to {args.out}")

    if review_log:
        pd.DataFrame(review_log).to_csv(args.review_out, index=False)
        logger.info(f"Wrote {len(review_log)} flagged bonds to {args.review_out}")


if __name__ == "__main__":
    main()