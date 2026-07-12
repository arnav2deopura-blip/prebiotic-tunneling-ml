import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. Your 12-molecule dataset, NOW WITH TARGET BDEs ADDED!
data = [
    {"name": "Formaldehyde", "smiles": "C=O", "class": "Aldehyde / Precursor", "target_bde_kjmol": 364},
    {"name": "Hydrogen Cyanide", "smiles": "C#N", "class": "Nitrile / Key Precursor", "target_bde_kjmol": 537},
    {"name": "Formamide", "smiles": "NC=O", "class": "Amide / Amino Acid Precursor", "target_bde_kjmol": 399},
    {"name": "Acetaldehyde", "smiles": "CC=O", "class": "Aldehyde / Precursor", "target_bde_kjmol": 374},
    {"name": "Methylamine", "smiles": "CN", "class": "Amine / Precursor", "target_bde_kjmol": 389},
    {"name": "Cyanoacetylene", "smiles": "C#CC#N", "class": "Nitrile / Precursor", "target_bde_kjmol": 556},
    {"name": "Glycolaldehyde", "smiles": "OCC=O", "class": "Sugar Precursor", "target_bde_kjmol": 365},
    {"name": "Glycine", "smiles": "NCC(=O)O", "class": "Simplest Amino Acid", "target_bde_kjmol": 350},
    {"name": "Benzene", "smiles": "c1ccccc1", "class": "Monocyclic Aromatic Precursor", "target_bde_kjmol": 473},
    {"name": "Naphthalene", "smiles": "c1ccc2ccccc2c1", "class": "2-Ring Simple PAH", "target_bde_kjmol": 470},
    {"name": "Anthracene", "smiles": "c1ccc2cc3ccccc3cc2c1", "class": "3-Ring Simple PAH", "target_bde_kjmol": 464},
    {"name": "Pyrene", "smiles": "c1cc2ccc3ccc4ccc1c5c2c3c45", "class": "4-Ring Simple PAH", "target_bde_kjmol": 465}
]

df = pd.DataFrame(data)

# 2. Function to extract Group A features
def extract_group_a_features(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    features = {
        "Exact_MW": Descriptors.ExactMolWt(mol),
        "Heavy_Atom_Count": Descriptors.HeavyAtomCount(mol),
        "Num_H_Donors": Descriptors.NumHDonors(mol),
        "Num_H_Acceptors": Descriptors.NumHAcceptors(mol),
        "Num_Rotatable_Bonds": Descriptors.NumRotatableBonds(mol),
        "Num_Aromatic_Rings": Descriptors.NumAromaticRings(mol)
    }
    return pd.Series(features)

print("⚡ Running RDKit feature extraction...")
feature_df = df['smiles'].apply(extract_group_a_features)
final_df = pd.concat([df, feature_df], axis=1)

# 3. Export to a CSV file (this will overwrite the old one with the new updated version)
output_filename = "astrochem_features_baseline.csv"
final_df.to_csv(output_filename, index=False)
print(f"Success! Updated dataset with BDE targets saved to {output_filename}")