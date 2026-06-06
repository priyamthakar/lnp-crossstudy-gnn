"""
Cross-validation split strategies for the LNP benchmark.

Leave-One-Study-Out (LOSO-CV) is the primary evaluation protocol:
each fold holds out one entire study as the test set, training on
all remaining studies. This tests genuine cross-study generalisation —
the scenario that matters for practical deployment.

We also include a scaffold split (Bemis-Murcko on ionizable lipid) and
a random split for comparison, following the ablation in Phase 1.
"""

import numpy as np
import pandas as pd
from typing import Iterator


def loso_splits(
    study_labels: pd.Series | np.ndarray,
    min_test_size: int = 5,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Yield one split dict per study with keys 'train', 'val', 'test'.

    Studies with fewer than min_test_size samples are skipped.
    val_fraction of training indices are held aside for early stopping.
    """
    labels = np.asarray(study_labels)
    unique_studies = pd.Series(labels).value_counts()
    usable = unique_studies[unique_studies >= min_test_size].index.tolist()

    for study in usable:
        test_idx = np.where(labels == study)[0]
        train_val_idx = np.where(labels != study)[0]

        if len(train_val_idx) < 20:
            continue

        rng = np.random.default_rng(seed + hash(study) % (2**32))
        perm = rng.permutation(len(train_val_idx))
        n_val = max(5, int(val_fraction * len(train_val_idx)))
        val_idx = train_val_idx[perm[:n_val]]
        train_idx = train_val_idx[perm[n_val:]]

        yield {
            "study": study,
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }


def random_split(
    n: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """Simple random train/val/test split by index."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def scaffold_split(
    smiles_series: pd.Series,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Bemis-Murcko scaffold split on ionizable lipid SMILES.
    Requires RDKit. Falls back to random split if RDKit unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        def get_scaffold(smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return smi
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

        scaffolds = smiles_series.map(get_scaffold)
        unique_scaffolds = scaffolds.unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_scaffolds)

        n_train_s = int(train_frac * len(unique_scaffolds))
        n_val_s = int(val_frac * len(unique_scaffolds))
        train_s = set(unique_scaffolds[:n_train_s])
        val_s = set(unique_scaffolds[n_train_s : n_train_s + n_val_s])

        train_idx = np.where(scaffolds.isin(train_s))[0]
        val_idx = np.where(scaffolds.isin(val_s))[0]
        test_idx = np.where(~scaffolds.isin(train_s | val_s))[0]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    except ImportError:
        print("RDKit not available — falling back to random split.")
        return random_split(len(smiles_series), train_frac, val_frac, seed)
