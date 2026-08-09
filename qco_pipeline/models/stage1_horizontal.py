"""
Stage 1 — Horizontal reduction model.

Per raw-circuit-node, 3-way classification over {KEEP, CANCEL, MERGE_INTO_PREV}
(see phase0/horizontal_labels.py for label derivation) plus a scalar angle
regression head used only for KEEP nodes that absorbed a merged rotation.

Architecture note on "dynamic structure": the GNN backbone has a FIXED number
of learned weight matrices (num_layers, each hidden_dim x hidden_dim), but it
is applied over a graph whose node/edge COUNT varies per circuit — so the
*computation* (how many message-passing hops actually fire, over how many
nodes) reshapes to whatever the input circuit's size is, while the parameter
count stays constant. This is what lets one trained model generalize across
circuits of any qubit count / gate count without retraining or padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from qco_pipeline.graph.circuit_graph import NODE_FEATURE_DIM
from qco_pipeline.phase0.horizontal_labels import Action


class Stage1HorizontalModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # concat=False keeps hidden_dim constant across layers regardless
            # of `heads`, so num_layers/hidden_dim can be tuned independently
            # of attention-head count.
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.action_head = nn.Linear(hidden_dim, len(Action))       # 3-way classification
        self.angle_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index)
            x = norm(x + residual)  # residual connection — stabilizes deep stacks on small graphs
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        action_logits = self.action_head(x)      # (num_nodes, 3)
        predicted_angle = self.angle_head(x).squeeze(-1)  # (num_nodes,)
        return action_logits, predicted_angle

    @torch.no_grad()
    def predict_actions(self, data: Data) -> torch.Tensor:
        self.eval()
        action_logits, _ = self.forward(data)
        return action_logits.argmax(dim=-1)
