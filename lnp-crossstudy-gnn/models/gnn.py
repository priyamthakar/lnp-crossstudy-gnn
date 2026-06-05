"""
LNP-MFGO: Multi-component GIN with Cross-Component Attention.

Architecture:
  Module A — separate GIN encoder per lipid (ionizable, helper, sterol, PEG)
  Module B — cross-component Transformer attention (inter-lipid interactions)
  Module C — per-study learnable embedding (lab-context conditioning)
  Module D — MLP prediction head fusing graph + tabular + study embeddings

Reference: Thakar et al. (2025), Cross-Study Evaluation of ML for LNP Property Prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ATOM_FEATURES_DIM = 33
BOND_FEATURES_DIM = 6
LIPID_NAMES = ["ionizable", "helper", "sterol", "peg"]


# ---------------------------------------------------------------------------
# Module A: GIN Layer and Encoder
# ---------------------------------------------------------------------------

class GINLayer(nn.Module):
    """Single GIN message-passing layer (Xu et al., 2019)."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, batch_num_atoms):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row])
        return self.mlp((1 + self.eps) * x + agg)


class GINEncoder(nn.Module):
    """Stack of GIN layers followed by global mean pooling."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURES_DIM,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINLayer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch_num_atoms):
        """
        Args:
            x: [total_atoms, atom_dim]
            edge_index: [2, total_edges]
            batch_num_atoms: list[int], atom count per graph in batch
        Returns:
            graph_emb: [batch_size, out_dim]
        """
        h = self.atom_embed(x)
        for layer in self.layers:
            h = layer(h, edge_index, batch_num_atoms)
            h = self.dropout(h)

        graph_embs, offset = [], 0
        for n in batch_num_atoms:
            graph_embs.append(h[offset : offset + n].mean(dim=0))
            offset += n
        return self.out_proj(torch.stack(graph_embs))


# ---------------------------------------------------------------------------
# Module B: Cross-Component Attention
# ---------------------------------------------------------------------------

class CrossComponentAttention(nn.Module):
    """
    One-layer Transformer encoder attending across the 4 lipid embeddings.
    Returns both the fused formulation embedding and the attention weights
    (the weights are used for interpretability / Figure 1 in the paper).
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, lipid_embeds):
        """
        Args:
            lipid_embeds: [batch, 4, embed_dim]
        Returns:
            formulation_emb: [batch, embed_dim]
            attn_weights:    [batch, 4, 4]
        """
        attn_out, attn_weights = self.attn(lipid_embeds, lipid_embeds, lipid_embeds)
        x = self.norm1(lipid_embeds + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1), attn_weights


# ---------------------------------------------------------------------------
# Full Model: LNP-MFGO
# ---------------------------------------------------------------------------

class LNPMFGO(nn.Module):
    """
    Full LNP-MFGO model: 4× GIN encoders → cross-component attention →
    fuse with tabular features and study embedding → scalar prediction.
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURES_DIM,
        hidden_dim: int = 128,
        n_gin_layers: int = 3,
        tabular_dim: int = 8,
        n_studies: int = 100,
        study_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Module A: one GIN encoder per lipid type
        self.gin_encoders = nn.ModuleDict(
            {
                name: GINEncoder(atom_dim, hidden_dim, hidden_dim, n_gin_layers, dropout)
                for name in LIPID_NAMES
            }
        )

        # Module B: cross-component attention
        self.cross_attn = CrossComponentAttention(hidden_dim, n_heads=4, dropout=dropout)

        # Module C: study embedding (+1 slot for unseen studies)
        self.study_embedding = nn.Embedding(n_studies + 1, study_emb_dim)

        # Tabular feature encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )

        # Module D: prediction head
        fusion_dim = hidden_dim + 32 + study_emb_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, graph_batch, tabular, study_ids):
        """
        Args:
            graph_batch: dict mapping lipid name → (x, edge_index, batch_num_atoms)
            tabular:     [batch, tabular_dim]  float tensor
            study_ids:   [batch]               long tensor
        Returns:
            pred:         [batch]       scalar predictions
            attn_weights: [batch, 4, 4] cross-component attention for interpretability
        """
        lipid_embs = []
        for name in LIPID_NAMES:
            x, ei, bna = graph_batch[name]
            lipid_embs.append(self.gin_encoders[name](x, ei, bna))

        lipid_stack = torch.stack(lipid_embs, dim=1)  # [B, 4, hidden_dim]
        formulation_emb, attn_weights = self.cross_attn(lipid_stack)

        tab_emb = self.tabular_encoder(tabular)
        study_emb = self.study_embedding(study_ids)

        fused = torch.cat([formulation_emb, tab_emb, study_emb], dim=-1)
        pred = self.predictor(fused).squeeze(-1)
        return pred, attn_weights

    def init_unknown_study_embedding(self, known_study_ids=None):
        """Set the unknown-study slot to the mean of all training study embeddings."""
        n = self.study_embedding.num_embeddings - 1  # last slot = unknown
        with torch.no_grad():
            if known_study_ids is not None:
                mean_emb = self.study_embedding.weight[known_study_ids].mean(dim=0)
            else:
                mean_emb = self.study_embedding.weight[:n].mean(dim=0)
            self.study_embedding.weight[n] = mean_emb
