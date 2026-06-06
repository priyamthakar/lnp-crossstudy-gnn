"""
Data loading, Z-normalisation, and PyTorch Dataset/DataLoader construction.

Z-normalisation: for each study, subtract the study mean and divide by the
study standard deviation of the target. This removes lab-level batch effects
and is the key preprocessing step that enables cross-study comparison.
Studies with fewer than 2 non-null target values fall back to the global std.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from preprocessing.mol_features import build_graph_cache, smiles_to_graph

LIPID_SMILES_COLS = [
    "ionizable_lipid_smiles",
    "helper_lipid_smiles",
    "sterol_lipid_smiles",
    "peg_lipid_smiles",
]
LIPID_NAMES = ["ionizable", "helper", "sterol", "peg"]

TABULAR_COLS = [
    "Ratio_Ionizable",
    "Ratio_Helper",
    "Ratio_Sterol",
    "Ratio_PEG",
    "Process_FlowRate",
    "Process_Ratio_AqOrg",
    "Process_Is_Microfluidic",
    "Process_pH",
]


def add_znorm_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Add a '<target_col>_znorm' column using per-study Z-normalisation.
    Modifies df in-place and returns it.
    """
    mask = df[target_col].notna()
    stats = df[mask].groupby("paper_doi")[target_col].agg(["mean", "std"])
    global_std = df[mask][target_col].std()
    stats["std"] = stats["std"].fillna(global_std)
    stats.loc[stats["std"] == 0, "std"] = global_std

    df["_study_mean"] = df["paper_doi"].map(stats["mean"]).fillna(df[mask][target_col].mean())
    df["_study_std"] = df["paper_doi"].map(stats["std"]).fillna(global_std)
    df[target_col + "_znorm"] = (df[target_col] - df["_study_mean"]) / df["_study_std"]
    df.drop(columns=["_study_mean", "_study_std"], inplace=True)
    return df


def load_and_featurize(
    csv_path: str,
    target_col: str = "particle_size_nm_std_num",
    encoding: str = "latin-1",
) -> tuple:
    """
    Load the LNP Atlas CSV, build molecular graphs, apply Z-normalisation.

    Returns:
        df          — filtered DataFrame (rows with all 4 SMILES parseable)
        graphs      — dict {lipid_name: [graph_dict, ...]} aligned to df rows
        tabular     — np.ndarray [n_rows, len(TABULAR_COLS)]
        study_ids   — np.ndarray [n_rows] integer study indices
        n_studies   — int, number of unique studies
    """
    df_raw = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
    df_raw = add_znorm_target(df_raw, target_col)

    # Build graph cache (one graph per unique SMILES)
    cache = {}
    for col in LIPID_SMILES_COLS:
        cache.update(build_graph_cache(df_raw[col]))

    # Filter to rows where all four SMILES parse correctly
    valid_rows, valid_graphs = [], {n: [] for n in LIPID_NAMES}
    for idx, row in df_raw.iterrows():
        graphs_row = {}
        ok = True
        for name, col in zip(LIPID_NAMES, LIPID_SMILES_COLS):
            g = cache.get(row[col])
            if g is None:
                ok = False
                break
            graphs_row[name] = g
        if ok:
            valid_rows.append(idx)
            for name in LIPID_NAMES:
                valid_graphs[name].append(graphs_row[name])

    df = df_raw.loc[valid_rows].reset_index(drop=True)

    # Encode study IDs
    unique_studies = df["paper_doi"].fillna("Unknown").unique()
    study_map = {s: i for i, s in enumerate(unique_studies)}
    study_ids = df["paper_doi"].fillna("Unknown").map(study_map).values.astype(int)

    tabular = df[TABULAR_COLS].fillna(df[TABULAR_COLS].median()).values.astype(np.float32)

    return df, valid_graphs, tabular, study_ids, len(unique_studies)


# ---------------------------------------------------------------------------
# PyTorch Dataset and custom collate
# ---------------------------------------------------------------------------

class LNPGraphDataset(Dataset):
    def __init__(self, indices, graphs_dict, tabular, targets, study_ids):
        self.indices = indices
        self.graphs = graphs_dict
        self.tabular = tabular
        self.targets = targets
        self.study_ids = study_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            {name: self.graphs[name][i] for name in LIPID_NAMES},
            self.tabular[i],
            self.targets[i],
            self.study_ids[i],
        )


def collate_lnp(batch):
    """Batch molecular graphs by concatenating atoms and offsetting edge indices."""
    graphs_list, tab_list, target_list, study_list = zip(*batch)

    graph_batch = {}
    for name in LIPID_NAMES:
        all_x, all_ei, batch_num_atoms = [], [], []
        offset = 0
        for sample in graphs_list:
            g = sample[name]
            all_x.append(g["x"])
            all_ei.append(g["edge_index"] + offset)
            batch_num_atoms.append(g["num_atoms"])
            offset += g["num_atoms"]
        graph_batch[name] = (
            torch.cat(all_x, dim=0),
            torch.cat(all_ei, dim=1),
            batch_num_atoms,
        )

    return (
        graph_batch,
        torch.FloatTensor(np.array(tab_list)),
        torch.FloatTensor(np.array(target_list)),
        torch.LongTensor(np.array(study_list)),
    )


def make_loaders(
    split_indices: dict,
    graphs_dict: dict,
    tabular: np.ndarray,
    targets: np.ndarray,
    study_ids: np.ndarray,
    batch_size: int = 32,
) -> tuple:
    """
    Create (train_loader, val_loader, test_loader) for one LOSO fold.
    Also fits a StandardScaler on training tabular features.

    split_indices: {'train': [...], 'val': [...], 'test': [...]}
    """
    scaler = StandardScaler()
    tab_train = scaler.fit_transform(tabular[split_indices["train"]])
    tab_val = scaler.transform(tabular[split_indices["val"]])
    tab_test = scaler.transform(tabular[split_indices["test"]])

    def _subset_graphs(idx_list):
        return {name: [graphs_dict[name][i] for i in idx_list] for name in LIPID_NAMES}

    def _ds(idx_list, tab_scaled):
        local_idx = np.arange(len(idx_list))
        return LNPGraphDataset(
            local_idx,
            _subset_graphs(idx_list),
            tab_scaled,
            targets[idx_list],
            study_ids[idx_list],
        )

    train_ds = _ds(split_indices["train"], tab_train)
    val_ds = _ds(split_indices["val"], tab_val)
    test_ds = _ds(split_indices["test"], tab_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_lnp, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_lnp)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_lnp)

    return train_loader, val_loader, test_loader, scaler
