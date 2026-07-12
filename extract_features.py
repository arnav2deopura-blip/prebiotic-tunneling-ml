import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. Your 12-molecule dataset
data = [
    {"name": "Formaldehyde", "smiles": "C=O", "class": "Aldehyde / Precursor"},
    {"name": "Hydrogen Cyanide", "smiles": "C#N", "class": "Nitrile / Key Precursor"},
    {"name": "Formamide", "smiles": "NC=O", "class": "Amide / Amino Acid Precursor"},
    {"name": "Acetaldehyde", "smiles": "CC=O", "class": "Aldehyde / Precursor"},
    {"name": "Methylamine", "smiles": "CN", "class": "Amine / Precursor"},
    {"name": "Cyanoacetylene", "smiles": "C#CC#N", "class": "Nitrile / Precursor"},
    {"name": "Glycolaldehyde", "smiles": "OCC=O", "class": "Sugar Precursor"},
    {"name": "Glycine", "smiles": "NCC(=O)O", "class": "Simplest Amino Acid"},
    {"name": "Benzene", "smiles": "c1ccccc1", "class": "Monocyclic Aromatic Precursor"},
    {"name": "Naphthalene", "smiles": "c1ccc2ccccc2c1", "class": "2-Ring Simple PAH"},
    {"name": "Anthracene", "smiles": "c1ccc2cc3ccccc3cc2c1", "class": "3-Ring Simple PAH"},
    {"name": "Pyrene", "smiles": "c1cc2ccc3ccc4ccc1c5c2c3c45", "class": "4-Ring Simple PAH"}
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

# 3. Export to a CSV file so you can open it in Excel, Google Sheets, or Notion!
output_filename = "astrochem_features_baseline.csv"
final_df.to_csv(output_filename, index=False)
print(f"✅ Success! Data saved to {output_filename}")