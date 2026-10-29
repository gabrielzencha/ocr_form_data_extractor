"""GeoLinker: text-free entity-level key->value linking.

Nodes are entities (gold in the standard RE protocol). Node input is the normalised box,
its size, and the gold role one-hot (the same information LiLT's RE decoder receives via its
entity-label embedding). Optional visual features are appended by the caller.
Message passing runs over a k-NN graph; every question x answer pair is scored, so
candidate coverage is 100%, exactly like the LiLT/LayoutXLM decoder.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat import EGATConv
from .graph import knn_edges

ROLE = {"header": 0, "question": 1, "answer": 2, "other": 3}


def pair_features(b: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    """Rich directed geometry for u->v: corner/centre offsets, gaps, overlaps, log-distance."""
    u, v = b[ei[0]], b[ei[1]]
    cu = torch.stack([(u[:, 0] + u[:, 2]) / 2, (u[:, 1] + u[:, 3]) / 2], 1)
    cv = torch.stack([(v[:, 0] + v[:, 2]) / 2, (v[:, 1] + v[:, 3]) / 2], 1)
    hx = (torch.minimum(u[:, 2], v[:, 2]) - torch.maximum(u[:, 0], v[:, 0]))
    vy = (torch.minimum(u[:, 3], v[:, 3]) - torch.maximum(u[:, 1], v[:, 1]))
    hu, hv = (u[:, 3] - u[:, 1]).clamp_min(1e-4), (v[:, 3] - v[:, 1]).clamp_min(1e-4)
    wu, wv = (u[:, 2] - u[:, 0]).clamp_min(1e-4), (v[:, 2] - v[:, 0]).clamp_min(1e-4)
    d = cv - cu
    return torch.cat([
        v[:, :2] - u[:, :2], v[:, 2:] - u[:, 2:],          # corner offsets
        (v[:, 0] - u[:, 2]).unsqueeze(1), (v[:, 1] - u[:, 3]).unsqueeze(1),  # gaps right / below
        d, torch.log1p(d.norm(dim=1, keepdim=True) * 10),
        (hx / torch.minimum(wu, wv)).unsqueeze(1), (vy / torch.minimum(hu, hv)).unsqueeze(1),
        torch.log(hv / hu).unsqueeze(1), torch.log(wv / wu).unsqueeze(1),
    ], 1)


PAIR_DIM = 13


def node_input(doc, ents, use_roles=True) -> torch.Tensor:
    b = torch.tensor([doc.norm_box(e.box) for e in ents], dtype=torch.float32).reshape(-1, 4)
    size = torch.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    feats = [b, size]
    if use_roles:
        feats.append(F.one_hot(torch.tensor([ROLE[e.label] for e in ents]), 4).float())
    return torch.cat(feats, 1)


def build_entity_graph(doc, k=8, use_roles=True):
    ents = doc.entities
    b = torch.tensor([doc.norm_box(e.box) for e in ents], dtype=torch.float32).reshape(-1, 4)
    ei = torch.from_numpy(knn_edges(b.numpy(), doc.width / doc.height, k))
    q = [i for i, e in enumerate(ents) if e.label == "question"]
    a = [i for i, e in enumerate(ents) if e.label == "answer"]
    cand = torch.tensor([(i, j) for i in q for j in a], dtype=torch.long).reshape(-1, 2).T
    gold = doc.kv_links
    y = torch.tensor([float((int(i), int(j)) in gold) for i, j in cand.T], dtype=torch.float32)
    return {"doc": doc, "x": node_input(doc, ents, use_roles), "box": b, "ei": ei,
            "ef": pair_features(b, ei) if ei.numel() else torch.zeros(0, PAIR_DIM),
            "cand": cand, "y": y}


class GeoLinker(nn.Module):
    def __init__(self, in_node, hidden=64, heads=3, layers=2, extra_node=0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_node + extra_node, hidden), nn.ReLU())
        self.convs = nn.ModuleList([EGATConv(hidden, PAIR_DIM, hidden, hidden, heads) for _ in range(layers)])
        self.mix = nn.ModuleList([nn.Linear(hidden * heads, hidden) for _ in range(layers)])
        self.score = nn.Sequential(nn.Linear(2 * hidden + PAIR_DIM, 128), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, g, extra=None):
        x = g["x"] if extra is None else torch.cat([g["x"], extra], 1)
        h = self.inp(x)
        if g["ei"].numel():
            for conv, mix in zip(self.convs, self.mix):
                m, _ = conv(h, g["ei"], g["ef"])
                h = F.relu(h + mix(m.flatten(1)))
        return h

    def forward(self, g, extra=None):
        h = self.encode(g, extra)
        if g["cand"].numel() == 0:
            return torch.zeros(0)
        i, j = g["cand"]
        return self.score(torch.cat([h[i], h[j], pair_features(g["box"], g["cand"])], 1)).squeeze(1)
