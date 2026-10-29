"""Geometric rule baseline for key->value linking with gold entities.

For each answer entity, choose the nearest question entity that is either to its left
with vertical overlap (same line) or above it with horizontal overlap. Distance is the
gap between boxes, normalised by page size. Uses no text.
"""
from __future__ import annotations


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def link_doc(doc, max_gap: float = 0.25) -> set[tuple[int, int]]:
    qs = [e for e in doc.entities if e.label == "question"]
    links = set()
    for a in doc.entities:
        if a.label != "answer":
            continue
        ax0, ay0, ax1, ay1 = doc.norm_box(a.box)
        best, best_d = None, max_gap
        for q in qs:
            qx0, qy0, qx1, qy1 = doc.norm_box(q.box)
            v_ov = _overlap(ay0, ay1, qy0, qy1) / max(1e-6, min(ay1 - ay0, qy1 - qy0))
            h_ov = _overlap(ax0, ax1, qx0, qx1) / max(1e-6, min(ax1 - ax0, qx1 - qx0))
            d = None
            if v_ov > 0.3 and qx1 <= ax0 + 0.01:      # same line, question on the left
                d = ax0 - qx1
            elif h_ov > 0.0 and qy1 <= ay0 + 0.005:   # question directly above
                d = (ay0 - qy1) * 1.5                 # prefer same-line keys
            if d is not None and d < best_d:
                best, best_d = q.id, d
        if best is not None:
            links.add((best, a.id))
    return links
