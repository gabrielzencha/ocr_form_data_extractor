"""Gold-free word graphs.

Candidate edges come only from geometry: each word is connected to its k nearest words by
box gap (aspect-corrected), and the union is made symmetric. Labels are attached after
construction from entity membership, so the test graph never encodes gold structure
(unlike LAGNN's released files; see REPLICATION.md).
"""
from __future__ import annotations

import numpy as np
import torch

EDGE_CLASSES = ("no-link", "same-entity", "key-value", "header-key")
NODE_CLASSES = ("header", "question", "answer", "other")


def norm_boxes(doc) -> np.ndarray:
    return np.array([doc.norm_box(w.box) for w in doc.words], dtype=np.float32).reshape(-1, 4)


def gap_distance(b: np.ndarray, aspect: float) -> np.ndarray:
    """Pairwise distance between box edges; x scaled by page width/height to be isotropic."""
    x0, y0, x1, y1 = (b[:, i] for i in range(4))
    dx = np.maximum(0, np.maximum(x0[None] - x1[:, None], x0[:, None] - x1[None])) * aspect
    dy = np.maximum(0, np.maximum(y0[None] - y1[:, None], y0[:, None] - y1[None]))
    return np.sqrt(dx ** 2 + dy ** 2)


def knn_edges(b: np.ndarray, aspect: float, k: int) -> np.ndarray:
    n = len(b)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    d = gap_distance(b, aspect)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nb = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    pairs = {(i, int(j)) for i in range(n) for j in nb[i]}
    pairs |= {(j, i) for i, j in pairs}
    return np.array(sorted(pairs), dtype=np.int64).T


def edge_features(b: np.ndarray, ei: np.ndarray) -> np.ndarray:
    """LAGNN-style relative spacing (6-D) for directed edges u->v."""
    u, v = b[ei[0]], b[ei[1]]
    return np.stack([v[:, 2] - u[:, 0], v[:, 0] - u[:, 0], v[:, 0] - u[:, 2],
                     v[:, 1] - u[:, 1], v[:, 3] - u[:, 3], v[:, 3] - u[:, 1]], 1).astype(np.float32)


def edge_labels(doc, ei: np.ndarray) -> np.ndarray:
    ent = np.array([w.entity for w in doc.words])
    kv = {frozenset(p) for p in doc.kv_links}
    hk = {frozenset((a, b)) for a, b in doc.links
          if doc.entities[a].label == "header" and doc.entities[b].label == "question"}
    y = np.zeros(ei.shape[1], dtype=np.int64)
    for i, (u, v) in enumerate(ei.T):
        eu, ev = ent[u], ent[v]
        if eu == ev:
            y[i] = 1
        elif frozenset((eu, ev)) in kv:
            y[i] = 2
        elif frozenset((eu, ev)) in hk:
            y[i] = 3
    return y


def build_graph(doc, k: int = 10) -> dict:
    b = norm_boxes(doc)
    ei = knn_edges(b, doc.width / doc.height, k)
    return {
        "doc": doc,
        "x": torch.from_numpy(b),
        "ei": torch.from_numpy(ei),
        "ef": torch.from_numpy(edge_features(b, ei)),
        "y_edge": torch.from_numpy(edge_labels(doc, ei)),
        "y_node": torch.tensor([NODE_CLASSES.index(doc.entities[w.entity].label) for w in doc.words]),
    }


def coverage(doc, ei: np.ndarray) -> tuple[int, int]:
    """(# gold key-value entity pairs with >=1 candidate word edge, # gold pairs)."""
    ent = np.array([w.entity for w in doc.words])
    seen = {(ent[u], ent[v]) for u, v in ei.T}
    gold = doc.kv_links
    return sum((a, b) in seen for a, b in gold), len(gold)
