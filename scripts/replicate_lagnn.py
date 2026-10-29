"""Replicate LAGNN (Voutharoja et al., ICDAR 2023) on the authors' released edge files.

Reports the paper's metric (macro-F1 over 6 word-pair edge classes) for FUNSD test and
zero-shot XFUND, using the same recipe as third_party/LAGNN/main.py. Predictions are
saved so scripts/eval_lagnn_entity_level.py can rescore them on the standard RE metric.

Deviation from the released code (documented in REPLICATION.md): the files label
same-entity pairs 'word-word', which the released class map does not handle; we map it
to 'same-entity' as the paper describes.
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formkv.egat import EGAT  # noqa: E402
from formkv.metrics import edge_macro_f1  # noqa: E402

CLASSES = ["question-answer", "header-question", "proximate_H", "same-entity", "proximate_V", "no-link"]
REMAP = {"question-header": "proximate_H", "header-header": "proximate_H", "question-question": "proximate_H",
         "question-other": "question-answer", "line-line": "proximate_V", "word-word": "same-entity"}
C2I = {c: i for i, c in enumerate(CLASSES)}


def load_split(d):
    graphs = []
    for fn in sorted(os.listdir(d)):
        lines = [l for l in open(os.path.join(d, fn)).read().split("\n") if l]
        if not lines:
            continue
        boxes, b2i, src, dst, y, ef, texts = [], {}, [], [], [], [], []
        for l in lines:
            c = l.split("\t")
            lab = REMAP.get(c[4], c[4])
            pair = []
            for t, b in ((c[0], c[2]), (c[1], c[3])):
                key = b  # original dedups nodes by box value
                if key not in b2i:
                    b2i[key] = len(boxes)
                    boxes.append([float(v) for v in b[1:-1].split(", ")])
                    texts.append(t)
                pair.append(b2i[key])
            src.append(pair[0]); dst.append(pair[1])
            y.append(C2I[lab]); ef.append([float(v) for v in c[-7:-1]])
        graphs.append({"file": fn, "x": torch.tensor(boxes), "ei": torch.tensor([src, dst]),
                       "ef": torch.tensor(ef), "y": torch.tensor(y), "texts": texts})
    return graphs


def run_eval(model, graphs):
    model.eval(); yt, yp, probs = [], [], []
    with torch.no_grad():
        for g in graphs:
            p = F.softmax(model(g["x"], g["ei"], g["ef"]), 1)
            probs.append(p.numpy()); yt += g["y"].tolist(); yp += p.argmax(1).tolist()
    return edge_macro_f1(yt, yp), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lagnn_dir", default="third_party/LAGNN/FUNSD")
    ap.add_argument("--out", default="results/lagnn_replication")
    ap.add_argument("--seed", type=int, default=6222)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--early_stop", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    tr = load_split(f"{a.lagnn_dir}/en/train"); va = load_split(f"{a.lagnn_dir}/en/val")
    model = EGAT()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bad, ck = -1, 0, f"{a.out}/seed{a.seed}.pt"
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for g in tr:  # batch size 1, as in the original
            opt.zero_grad()
            F.cross_entropy(model(g["x"], g["ei"], g["ef"]), g["y"]).backward()
            opt.step()
        f, _ = run_eval(model, va)
        if f > best:
            best, bad = f, 0; torch.save(model.state_dict(), ck)
        else:
            bad += 1
        if ep % 25 == 0:
            print(f"ep {ep} val_macroF1 {f:.4f} best {best:.4f} {time.time()-t0:.0f}s", flush=True)
        if bad == a.early_stop:
            print(f"early stop at ep {ep}", flush=True); break
    model.load_state_dict(torch.load(ck))

    res = {"seed": a.seed, "params": nparam, "best_val_macroF1": best, "epochs_run": ep + 1, "test": {}}
    for lang in sorted(os.listdir(a.lagnn_dir)):
        gs = load_split(f"{a.lagnn_dir}/{lang}/test")
        f, probs = run_eval(model, gs)
        res["test"][lang] = round(f, 4)
        np.savez_compressed(f"{a.out}/pred_{lang}_seed{a.seed}.npz",
                            **{g["file"]: p for g, p in zip(gs, probs)})
        print(f"TEST {lang}: edge macro-F1 = {f:.4f}", flush=True)
    json.dump(res, open(f"{a.out}/result_seed{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
