"""
Stage 2 — Vertical reduction model.

Unlike Stage 1 (per-gate-node classification), Stage 2 makes a per-QUBIT
decision: prune or keep. The circuit graph has no explicit "qubit nodes", so
we pool gate-node embeddings onto their owning wire(s) (mean pool over every
node that touches a given qubit index) and classify each pooled wire vector.

This keeps the same "reshape to input size" property as Stage 1: the pooling
step produces exactly n_qubits wire-vectors for an n-qubit circuit, whatever n
is, using the same fixed-size backbone weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from qco_pipeline.graph.circuit_graph import NODE_FEATURE_DIM


def pool_nodes_to_wires(node_embeddings: torch.Tensor, node_qubits: list[list[int]], n_qubits: int) -> torch.Tensor:
    """Mean-pool node embeddings onto each qubit wire they touch."""
    hidden_dim = node_embeddings.size(-1)
    wire_sum = torch.zeros(n_qubits, hidden_dim, device=node_embeddings.device)
    wire_count = torch.zeros(n_qubits, device=node_embeddings.device)

    for node_id, qubits in enumerate(node_qubits):
        for q in qubits:
            wire_sum[q] += node_embeddings[node_id]
            wire_count[q] += 1

    wire_count = wire_count.clamp(min=1.0).unsqueeze(-1)
    return wire_sum / wire_count  # (n_qubits, hidden_dim); idle wires => zero vector


class Stage2VerticalModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout

        # Wire classifier also sees a normalized "activity" scalar (fraction
        # of total gates touching this wire) — a cheap, strong prior signal
        # that an idle/near-idle wire is a prune candidate.
        self.wire_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # single logit: prune probability
        )

    def encode_nodes(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index)
            x = norm(x + residual)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data: Data, node_qubits: list[list[int]], n_qubits: int) -> torch.Tensor:
        node_embeddings = self.encode_nodes(data)
        wire_embeddings = pool_nodes_to_wires(node_embeddings, node_qubits, n_qubits)

        activity = torch.zeros(n_qubits, 1, device=wire_embeddings.device)
        total_touches = max(sum(len(q) for q in node_qubits), 1)
        for qubits in node_qubits:
            for q in qubits:
                activity[q, 0] += 1.0 / total_touches

        wire_input = torch.cat([wire_embeddings, activity], dim=-1)
        prune_logit = self.wire_head(wire_input).squeeze(-1)  # (n_qubits,)
        return prune_logit

    @torch.no_grad()
    def predict_keep_mask(self, data: Data, node_qubits: list[list[int]], n_qubits: int, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        prune_logit = self.forward(data, node_qubits, n_qubits)
        prune_prob = torch.sigmoid(prune_logit)
        return prune_prob < threshold  # True = keep
