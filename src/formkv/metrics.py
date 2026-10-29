"""Evaluation metrics.

link_prf        Standard RE metric of LayoutLMv2 / LayoutXLM / LiLT: gold entities are
                given, score directed question->answer links pooled over documents.
pair_prf        End-to-end pair extraction (PEneo-style): a predicted pair counts if its
                key and value word sets both exactly match a gold pair.
edge_macro_f1   LAGNN's reported number: macro-F1 over edge classes on candidate word pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import f1_score


@dataclass
class PRF:
    tp: int
    n_pred: int
    n_gold: int

    @property
    def p(self):
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def r(self):
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self):
        return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0

    def as_dict(self):
        return {"p": round(self.p, 4), "r": round(self.r, 4), "f1": round(self.f1, 4),
                "tp": self.tp, "n_pred": self.n_pred, "n_gold": self.n_gold}


def link_prf(preds: list[set], golds: list[set]) -> PRF:
    tp = sum(len(p & g) for p, g in zip(preds, golds))
    return PRF(tp, sum(len(p) for p in preds), sum(len(g) for g in golds))


def pair_prf(preds: list[set], golds: list[set]) -> PRF:
    """Each element is a set of (frozenset(key_word_ids), frozenset(value_word_ids))."""
    return link_prf(preds, golds)


def gold_pairs(doc) -> set:
    return {(frozenset(doc.entities[a].words), frozenset(doc.entities[b].words)) for a, b in doc.kv_links}


def edge_macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))
