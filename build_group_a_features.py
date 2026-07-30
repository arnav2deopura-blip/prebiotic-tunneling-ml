import argparse
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "P"})
MAX_HEAVY_ATOMS = 30
REACTIVE_BOND_SMARTS = {
    "C-H": "[#6]-[#1]",
    "N-H": "[#7]-[#1]",
    "O-H": "[#8]-[#1]",
}

# These are bookkeeping symbols that KIDA/UMIST use inside reaction networks —
# not real molecules — so they should never be treated as "unresolved" species.
EXCLUDE_NAMES = frozenset({"CR", "CRP", "CRPHOT", "Photon", "PHOTON", "e-"})

# Species where the automatic InChI/PubChem route can't produce a usable SMILES,
# but the structure is well-established chemistry, so we hand-enter it here.
# (See chat notes for the reasoning behind each one.)
MANUAL_SMILES_OVERRIDES = {
    "CCl": "[C]Cl",              # neutral CCl radical, 3 unpaired electrons on carbon
    "CCl+": "[C+]Cl",            # cation form of the above
    "HCOOH+": "O=C[OH+]",        # formic-acid radical cation (unpaired electron on O-H oxygen)
    "C2H5+": "C[CH2+]",          # ethyl cation (classic, closed-shell carbocation)
    "C4H7+": "CC=C[CH2+]",       # allylic-type C4H7+ cation
    "C6H7+": "C1C=CC=C[CH+]1",   # benzenium ion (protonated benzene)
    "H3+": "[H][H+][H]",         # trihydrogen cation (localized approximation)
    "H2D+": "[H][H+][2H]",       # deuterated H3+
}


def _build_hypervalent_mol(atoms, bonds, charges, radicals):
    """Builds species like CH4+/CH5+ whose bonding (more bonds on an atom than its
    normal valence allows) can't survive a plain SMILES round-trip. We construct
    the molecule graph directly and skip only the valence-legality check."""
    rw = Chem.RWMol()
    for sym in atoms:
        rw.AddAtom(Chem.Atom(sym))
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, order in bonds:
        rw.AddBond(i, j, order_map[order])
    for idx, chg in charges.items():
        rw.GetAtomWithIdx(idx).SetFormalCharge(chg)
    for idx, rad in radicals.items():
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumRadicalElectrons(rad)
        atom.SetNoImplicit(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    return mol


MANUAL_HYPERVALENT_OVERRIDES = {
    # CH4+: carbon with 4 normal bonds to H, +1 charge, 1 radical electron (methane radical cation)
    "CH4+": lambda: _build_hypervalent_mol(
        ["C", "H", "H", "H", "H"], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)], {0: 1}, {0: 1}
    ),
    # CH5+: carbon with 5 bonds to H, +1 charge, 0 radical electrons (protonated methane)
    "CH5+": lambda: _build_hypervalent_mol(
        ["C", "H", "H", "H", "H", "H"], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1)], {0: 1}, {0: 0}
    ),
}
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/CanonicalSMILES,ConnectivitySMILES/json"

try:
    SSL_CONTEXT = ssl._create_unverified_context()
except AttributeError:
    SSL_CONTEXT = None


def load_kida(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"Species": "name", "Formula": "formula", "Inchi": "inchi"})
    df["inchi"] = df["inchi"].fillna("")
    df["identifier_type"] = df["inchi"].apply(
        lambda v: "inchi" if isinstance(v, str) and len(v.strip()) > len("InChI=") else "name_only"
    )
    df["identifier_value"] = df.apply(
        lambda r: r["inchi"] if r["identifier_type"] == "inchi" else r["name"], axis=1
    )
    df["source"] = "KIDA"
    return df[["name", "formula", "identifier_type", "identifier_value", "source"]]


def load_umist(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("species", "name"):
            rename_map[col] = "name"
        elif low == "formula":
            rename_map[col] = "formula"
        elif low in ("inchi", "inchi_string", "inchistring"):
            rename_map[col] = "inchi"
    df = df.rename(columns=rename_map)
    if "inchi" not in df.columns:
        df["inchi"] = ""
    if "formula" not in df.columns:
        df["formula"] = ""
    df["inchi"] = df["inchi"].fillna("")
    df["identifier_type"] = df["inchi"].apply(
        lambda v: "inchi" if isinstance(v, str) and len(v.strip()) > len("InChI=") else "name_only"
    )
    df["identifier_value"] = df.apply(
        lambda r: r["inchi"] if r["identifier_type"] == "inchi" else r["name"], axis=1
    )
    df["source"] = "UMIST"
    return df[["name", "formula", "identifier_type", "identifier_value", "source"]]


def load_curated(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "formula" not in df.columns:
        df["formula"] = ""
    if "source" not in df.columns:
        df["source"] = "curated"
    else:
        df["source"] = df["source"].fillna("curated")
    return df[["name", "formula", "identifier_type", "identifier_value", "source"]]


def load_pubchem_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_pubchem_checkpoint(path, resolved):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resolved, f)


def pubchem_bulk_resolve(names, checkpoint_path, chunk_size=100):
    resolved = load_pubchem_checkpoint(checkpoint_path)
    remaining = [n for n in names if n not in resolved]
    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i:i + chunk_size]
        post_data = "\n".join(chunk).encode("utf-8")
        req = urllib.request.Request(
            PUBCHEM_URL,
            data=post_data,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Content-Type": "text/plain",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
                properties = data.get("PropertyTable", {}).get("Properties", [])
                for idx, prop in enumerate(properties):
                    if idx < len(chunk):
                        smiles = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
                        if smiles:
                            resolved[chunk[idx]] = smiles
        except Exception:
            pass
        save_pubchem_checkpoint(checkpoint_path, resolved)
        time.sleep(1.0)
    return resolved


def mol_from_inchi(inchi):
    try:
        # KIDA's own export has at least one row with an uppercase "/Q+1" instead
        # of the InChI-standard lowercase "/q+1" (found in C7H+) — normalize it.
        cleaned = re.sub(r"/Q([+-])", r"/q\1", str(inchi).strip())
        return Chem.MolFromInchi(cleaned)
    except Exception:
        return None


def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def strip_charge_suffix(formula):
    return re.sub(r"[+-]\d*$", "", formula)


def element_set(formula):
    tokens = re.findall(r"[A-Z][a-z]?", strip_charge_suffix(formula))
    return {("H" if t == "D" else t) for t in tokens}


def formulas_roughly_match(computed_formula, stated_formula):
    if not isinstance(stated_formula, str) or not stated_formula.strip():
        return True
    return element_set(computed_formula) == element_set(stated_formula)


def passes_element_whitelist(mol):
    return all(atom.GetSymbol() in ALLOWED_ELEMENTS for atom in mol.GetAtoms())


def passes_size_limit(mol):
    return Descriptors.HeavyAtomCount(mol) <= MAX_HEAVY_ATOMS


def passes_reactive_bond_check(mol):
    mol_with_hs = Chem.AddHs(mol)
    for smarts in REACTIVE_BOND_SMARTS.values():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol_with_hs.HasSubstructMatch(pattern):
            return True
    return False


def has_oxygen(mol):
    return any(atom.GetSymbol() == "O" for atom in mol.GetAtoms())


def charge_category(net_charge):
    if net_charge > 0:
        return "positive"
    if net_charge < 0:
        return "negative"
    return "neutral"


def extract_group_a_descriptors(mol):
    return {
        "Exact_MW": Descriptors.ExactMolWt(mol),
        "Heavy_Atom_Count": Descriptors.HeavyAtomCount(mol),
        "Num_H_Donors": Descriptors.NumHDonors(mol),
        "Num_H_Acceptors": Descriptors.NumHAcceptors(mol),
        "Num_Rotatable_Bonds": Descriptors.NumRotatableBonds(mol),
        "Num_Aromatic_Rings": Descriptors.NumAromaticRings(mol),
        "has_oxygen": has_oxygen(mol),
    }


def resolve_all_sources(kida_path, umist_path, curated_path, checkpoint_path):
    frames = []
    if kida_path and os.path.exists(kida_path):
        frames.append(load_kida(kida_path))
    if umist_path and os.path.exists(umist_path):
        frames.append(load_umist(umist_path))
    if curated_path and os.path.exists(curated_path):
        frames.append(load_curated(curated_path))

    if not frames:
        raise ValueError("No input source files were found.")

    combined = pd.concat(frames, ignore_index=True)

    # Drop bookkeeping placeholders (cosmic rays, photons, free electrons — not real molecules)
    combined = combined[~combined["name"].isin(EXCLUDE_NAMES)].reset_index(drop=True)

    resolved_records = []
    unresolved_records = []
    formula_mismatch_records = []
    invalid_structure_records = []

    # Pull out rows we have a hand-checked SMILES for, before the automatic routes run
    override_names = set(MANUAL_SMILES_OVERRIDES) | set(MANUAL_HYPERVALENT_OVERRIDES)
    override_rows = combined[combined["name"].isin(override_names)].copy()
    combined = combined[~combined["name"].isin(override_names)].copy()

    for _, row in override_rows.iterrows():
        name = row["name"]
        if name in MANUAL_SMILES_OVERRIDES:
            mol = mol_from_smiles(MANUAL_SMILES_OVERRIDES[name])
        else:
            mol = MANUAL_HYPERVALENT_OVERRIDES[name]()
        if mol is None:
            unresolved_records.append({**row.to_dict(), "reason": "manual_override_failed"})
            continue
        resolved_records.append({
            "name": row["name"],
            "smiles": Chem.MolToSmiles(mol),
            "source": row["source"],
        })

    inchi_rows = combined[combined["identifier_type"] == "inchi"].copy()
    name_rows = combined[combined["identifier_type"] == "name_only"].copy()

    pubchem_names = sorted(set(name_rows["identifier_value"].astype(str).tolist()))
    resolved_names = pubchem_bulk_resolve(pubchem_names, checkpoint_path) if pubchem_names else {}

    for _, row in inchi_rows.iterrows():
        mol = mol_from_inchi(row["identifier_value"])
        if mol is None:
            unresolved_records.append({**row.to_dict(), "reason": "inchi_parse_failed"})
            continue
        computed_formula = rdMolDescriptors.CalcMolFormula(mol)
        if not formulas_roughly_match(computed_formula, row["formula"]):
            formula_mismatch_records.append({
                **row.to_dict(),
                "computed_formula": computed_formula,
            })
            continue
        canonical_smiles = Chem.MolToSmiles(mol)
        if "." in canonical_smiles:
            invalid_structure_records.append({**row.to_dict(), "reason": "disconnected_fragments", "smiles": canonical_smiles})
            continue
        resolved_records.append({
            "name": row["name"],
            "smiles": canonical_smiles,
            "source": row["source"],
        })

    for _, row in name_rows.iterrows():
        smiles = resolved_names.get(str(row["identifier_value"]))
        if not smiles:
            unresolved_records.append({**row.to_dict(), "reason": "pubchem_lookup_failed"})
            continue
        mol = mol_from_smiles(smiles)
        if mol is None:
            invalid_structure_records.append({**row.to_dict(), "reason": "smiles_parse_failed", "smiles": smiles})
            continue
        canonical_smiles = Chem.MolToSmiles(mol)
        if "." in canonical_smiles:
            invalid_structure_records.append({**row.to_dict(), "reason": "disconnected_fragments", "smiles": canonical_smiles})
            continue
        resolved_records.append({
            "name": row["name"],
            "smiles": canonical_smiles,
            "source": row["source"],
        })

    resolved_df = pd.DataFrame(resolved_records)
    unresolved_df = pd.DataFrame(unresolved_records)
    formula_mismatch_df = pd.DataFrame(formula_mismatch_records)
    invalid_structure_df = pd.DataFrame(invalid_structure_records)

    return resolved_df, unresolved_df, formula_mismatch_df, invalid_structure_df


def dedup_and_tag(resolved_df):
    if resolved_df.empty:
        return resolved_df

    merged = {}
    for _, row in resolved_df.iterrows():
        mol = mol_from_smiles(row["smiles"])
        if mol is None and row["name"] in MANUAL_HYPERVALENT_OVERRIDES:
            # Known non-classical species (e.g. CH4+, CH5+) can't survive a plain
            # SMILES re-parse — rebuild them the same special way instead of dropping them.
            mol = MANUAL_HYPERVALENT_OVERRIDES[row["name"]]()
        if mol is None:
            continue
        canonical_smiles = Chem.MolToSmiles(mol)
        key = canonical_smiles
        if key not in merged:
            merged[key] = {
                "name": row["name"],
                "canonical_smiles": canonical_smiles,
                "sources": set(),
            }
        merged[key]["sources"].add(row["source"])

    records = []
    for entry in merged.values():
        sources = entry["sources"]
        detected_in_ism = bool(sources & {"KIDA", "UMIST"})
        records.append({
            "name": entry["name"],
            "smiles": entry["canonical_smiles"],
            "source": ",".join(sorted(sources)),
            "detected_in_ism": detected_in_ism,
        })
    return pd.DataFrame(records)


def apply_structural_filters(deduped_df):
    kept_records = []
    rejected_records = []

    for _, row in deduped_df.iterrows():
        mol = mol_from_smiles(row["smiles"])
        if mol is None and row["name"] in MANUAL_HYPERVALENT_OVERRIDES:
            mol = MANUAL_HYPERVALENT_OVERRIDES[row["name"]]()
        if mol is None:
            rejected_records.append({**row.to_dict(), "reason": "smiles_parse_failed"})
            continue
        if not passes_element_whitelist(mol):
            rejected_records.append({**row.to_dict(), "reason": "element_whitelist"})
            continue
        if not passes_size_limit(mol):
            rejected_records.append({**row.to_dict(), "reason": "size_limit"})
            continue
        if not passes_reactive_bond_check(mol):
            rejected_records.append({**row.to_dict(), "reason": "reactive_bond"})
            continue

        net_charge = Chem.GetFormalCharge(mol)
        descriptors = extract_group_a_descriptors(mol)
        kept_records.append({
            **row.to_dict(),
            "molecular_charge": charge_category(net_charge),
            "net_formal_charge": net_charge,
            **descriptors,
        })

    return pd.DataFrame(kept_records), pd.DataFrame(rejected_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kida", default="kida_species.csv")
    parser.add_argument("--umist", default="umist_species.csv")
    parser.add_argument("--curated", default="curated_candidates.csv")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--checkpoint", default="pubchem_checkpoint.json")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    resolved_df, unresolved_df, formula_mismatch_df, invalid_structure_df = resolve_all_sources(
        args.kida, args.umist, args.curated, args.checkpoint
    )

    deduped_df = dedup_and_tag(resolved_df)
    final_df, rejected_df = apply_structural_filters(deduped_df)

    final_df.to_csv(os.path.join(args.outdir, "candidate_molecules_filtered.csv"), index=False)
    unresolved_df.to_csv(os.path.join(args.outdir, "needs_manual_smiles.csv"), index=False)
    formula_mismatch_df.to_csv(os.path.join(args.outdir, "flagged_formula_mismatch.csv"), index=False)
    invalid_structure_df.to_csv(os.path.join(args.outdir, "flagged_invalid_structure.csv"), index=False)
    rejected_df.to_csv(os.path.join(args.outdir, "rejected_by_structural_filter.csv"), index=False)

    print(f"Resolved: {len(resolved_df)}")
    print(f"Unresolved (needs manual SMILES): {len(unresolved_df)}")
    print(f"Flagged formula mismatches: {len(formula_mismatch_df)}")
    print(f"Flagged invalid structures: {len(invalid_structure_df)}")
    print(f"Deduplicated candidates: {len(deduped_df)}")
    print(f"Passed structural filters: {len(final_df)}")
    print(f"Rejected by structural filters: {len(rejected_df)}")


if __name__ == "__main__":
    main()
