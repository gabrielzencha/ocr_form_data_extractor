"""Pure-PyTorch port of LAGNN's EGAT (third_party/LAGNN/egat_model.py), no DGL.

Semantics match the original: edge update from [h_src | f | h_dst], per-head attention
scalars softmaxed over each node's incoming edges, nodes with no incoming edges get 0,
edge classifier on [h_u | h_v | e_uv].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` (E, H, 1) grouped by destination `index` (E,)."""
    H = scores.shape[1]
    idx = index.view(-1, 1, 1).expand_as(scores)
    mx = torch.full((num_nodes, H, 1), float("-inf"), device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[index])
    den = torch.zeros((num_nodes, H, 1), device=scores.device, dtype=scores.dtype).index_add_(0, index, ex)
    return ex / den[index].clamp_min(1e-16)


class EGATConv(nn.Module):
    def __init__(self, in_node, in_edge, out_node, out_edge, heads):
        super().__init__()
        self.h, self.on, self.oe = heads, out_node, out_edge
        self.fc_nodes = nn.Linear(in_node, out_node * heads, bias=True)
        self.fc_edges = nn.Linear(in_edge + 2 * in_node, out_edge * heads, bias=False)
        self.fc_attn = nn.Linear(out_edge, heads, bias=False)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_edges.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def forward(self, x, edge_index, ef):
        src, dst = edge_index
        f = F.leaky_relu(self.fc_edges(torch.cat([x[src], ef, x[dst]], dim=-1))).view(-1, self.h, self.oe)
        a = F.leaky_relu(self.fc_attn(f).sum(-1).unsqueeze(-1))  # (E, H, 1)
        alpha = segment_softmax(a, dst, x.shape[0])
        hn = self.fc_nodes(x).view(-1, self.h, self.on)
        out = torch.zeros((x.shape[0], self.h, self.on), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, alpha * hn[src])
        return out, f


class EGAT(nn.Module):
    def __init__(self, in_node=4, in_edge=6, hidden=64, heads=3, n_classes=6):
        super().__init__()
        self.layer1 = EGATConv(in_node, in_edge, hidden, hidden, heads)
        self.W = nn.Linear(hidden * 3 * heads, n_classes)

    def forward(self, x, edge_index, ef):
        h, e = self.layer1(x, edge_index, ef)
        src, dst = edge_index
        return self.W(torch.cat([h[src].flatten(1), h[dst].flatten(1), e.flatten(1)], dim=1))
