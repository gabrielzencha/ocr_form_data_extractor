"""Rescore LAGNN replication predictions on the standard RE metric (gold entities + roles).

A predicted 'question-answer' word edge (u, v) becomes an entity link between the entities
containing u and v when one is a question and the other an answer, oriented question->answer.
This is the same information the LayoutXLM/LiLT RE protocol gives its decoder.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from formkv.data import load  # noqa: E402
from formkv.metrics import link_prf  # noqa: E402
from replicate_lagnn import load_split, C2I  # noqa: E402

QA = C2I["question-answer"]
DS = {"en": "funsd", **{l: f"xfund-{l}" for l in ("ja", "zh", "fr", "pt", "es", "it")}}


def match_words(doc, g):
    """Map graph node idx -> doc word idx by nearest normalised box with the same text."""
    nb = np.array([doc.norm_box(w.box) for w in doc.words])
    out = {}
    for i, (box, text) in enumerate(zip(g["x"].numpy(), g["texts"])):
        d = np.abs(nb - box).sum(1)
        cand = [j for j in np.argsort(d)[:5] if doc.words[j].text == text] or [int(np.argmin(d))]
        if d[cand[0]] < 0.02:
            out[i] = cand[0]
    return out


res = {}
for seed in (6222, 1, 2):
    res[seed] = {}
    for lang, ds in DS.items():
        docs = {d.doc_id: d for d in load(ds, "test")}
        graphs = load_split(f"third_party/LAGNN/FUNSD/{lang}/test")
        P = np.load(f"results/lagnn_replication/pred_{lang}_seed{seed}.npz")
        preds, golds, unmatched, total = [], [], 0, 0
        for g in graphs:
            doc = docs[g["file"][:-4]]
            m = match_words(doc, g); total += len(g["texts"]); unmatched += len(g["texts"]) - len(m)
            yp = P[g["file"]].argmax(1)
            links = set()
            for (u, v), y in zip(g["ei"].T.tolist(), yp):
                if y != QA or u not in m or v not in m:
                    continue
                eu, ev = doc.words[m[u]].entity, doc.words[m[v]].entity
                lu, lv = doc.entities[eu].label, doc.entities[ev].label
                if lu == "question" and lv == "answer":
                    links.add((eu, ev))
                elif lu == "answer" and lv == "question":
                    links.add((ev, eu))
            preds.append(links); golds.append(doc.kv_links)
        missing = [d for d in docs if d + ".txt" not in {g["file"] for g in graphs}]
        for d in missing:  # documents absent from LAGNN's files count as zero predictions
            preds.append(set()); golds.append(docs[d].kv_links)
        r = link_prf(preds, golds)
        res[seed][ds] = {**r.as_dict(), "unmatched_nodes": unmatched, "nodes": total, "missing_docs": len(missing)}
        print(f"seed {seed} {ds:9s} entity-level RE F1 {r.f1:.3f} (P {r.p:.3f} R {r.r:.3f}) unmatched {unmatched}/{total} missing docs {len(missing)}")
json.dump(res, open("results/lagnn_replication/re_entity_level.json", "w"), indent=1)
