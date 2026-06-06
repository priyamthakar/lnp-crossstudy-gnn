"""
Molecular graph featurization: SMILES → atom/bond feature tensors.

Each LNP formulation contains four lipid components (ionizable, helper, sterol, PEG).
This module converts each SMILES string into a graph dict with atom features,
edge indices, and bond features ready for the GIN encoder.

Atom features (33-dim):
  atomic number one-hot (11), degree (7), formal charge (5),
  radical electrons (3), hybridization (6), aromaticity (1)

Bond features (6-dim):
  bond type one-hot (4), conjugation (1), ring membership (1)
"""

import pandas as pd
import numpy as np
import torch
from rdkit import Chem

ATOM_FEATURES_DIM = 33
BOND_FEATURES_DIM = 6

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # C N O F P S Cl Br I Other
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(val, allowed_set):
    enc = [0] * (len(allowed_set) + 1)
    if val in allowed_set:
        enc[allowed_set.index(val)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom) -> list:
    """33-dimensional atom feature vector."""
    f = []
    f += _one_hot(atom.GetAtomicNum(), ATOM_LIST)
    f += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    f += _one_hot(atom.GetFormalCharge(), [-1, 0, 1, 2])
    f += _one_hot(atom.GetNumRadicalElectrons(), [0, 1])
    f += _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
    f += [float(atom.GetIsAromatic())]
    return f


def bond_features(bond) -> list:
    """6-dimensional bond feature vector."""
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> dict | None:
    """
    Convert a SMILES string to a graph dict.

    Returns None if SMILES is invalid or empty.

    Returns:
        {
          'x':          FloatTensor [n_atoms, 33]
          'edge_index': LongTensor  [2, n_edges]
          'edge_attr':  FloatTensor [n_edges, 6]
          'num_atoms':  int
        }
    """
    if not isinstance(smiles, str) or pd.isna(smiles) or len(smiles) < 3:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = list(mol.GetAtoms())
    if not atoms:
        return None

    x = torch.FloatTensor([atom_features(a) for a in atoms])

    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_feats += [bf, bf]

    if edge_indices:
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_feats)
    else:
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.zeros(1, BOND_FEATURES_DIM)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "num_atoms": len(atoms)}


def build_graph_cache(smiles_series) -> dict:
    """Build a {smiles: graph_dict} cache to avoid re-computing identical molecules."""
    cache = {}
    for smiles in smiles_series.dropna().unique():
        if smiles not in cache:
            cache[smiles] = smiles_to_graph(smiles)
    return cache
