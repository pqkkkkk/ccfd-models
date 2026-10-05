"""Graph Neural Network Architectures for Fraud Detection.

Includes:
- GraphSAGE: Inductive representation learning via neighborhood aggregation.
- GAT: Graph Attention Network with multi-head attention over transactions.
- GCN: Graph Convolutional Network baseline.
- HeteroGNN: Bipartite GNN for User <-> Transaction interactions using to_hetero.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    HeteroConv,
    SAGEConv
)


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE model for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output / final conv layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits


class GAT(nn.Module):
    """Multi-layer Graph Attention Network with multi-head attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GAT layer with multi-head concatenation
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Intermediate layers
        current_dim = hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(current_dim, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final GAT layer (average over heads)
        if num_layers > 1:
            self.convs.append(GATConv(current_dim, hidden_channels, heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits


class GCN(nn.Module):
    """Graph Convolutional Network baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x).squeeze(-1)
        return logits


class HeteroBipartiteGNN(nn.Module):
    """Heterogeneous GNN for Bipartite User <-> Transaction Graph."""

    def __init__(
        self,
        user_in_dim: int,
        tx_in_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Linear projections to align feature dimensions
        self.user_proj = nn.Linear(user_in_dim, hidden_channels)
        self.tx_proj = nn.Linear(tx_in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("user", "performs", "transaction"): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels, aggr="mean"
                    ),
                    ("transaction", "performed_by", "user"): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels, aggr="mean"
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        # Project initial features
        x_dict = {
            "user": F.relu(self.user_proj(x_dict["user"])),
            "transaction": F.relu(self.tx_proj(x_dict["transaction"])),
        }

        # Heterogeneous message passing with residual skip connections
        for conv in self.convs:
            out_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.relu(out_dict[key] + x_dict[key]) for key in x_dict.keys()
            }
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Predict on transaction nodes
        tx_out = x_dict["transaction"]
        logits = self.classifier(tx_out).squeeze(-1)
        return logits
