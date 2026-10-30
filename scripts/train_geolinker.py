"""Train GeoLinker (text-free) on FUNSD train; evaluate RE (gold entities + roles) on FUNSD
test and zero-shot on all XFUND test sets. Several seeds; reports mean/std.

Decoding: a pair is predicted if its score > 0.5. Optional --one_key_per_value keeps only
the highest-scoring key for each value (answers in FUNSD/XFUND almost always have one key).
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formkv.data import load, XFUND_LANGS  # noqa: E402
from formkv.geolinker import GeoLinker, build_entity_graph  # noqa: E402
from formkv.metrics import link_prf  # noqa: E402

EVAL = ["funsd"] + [f"xfund-{l}" for l in XFUND_LANGS]


def decode(g, s, one_key):
    p = torch.sigmoid(s)
    pairs = set()
    if one_key:
        best = {}
        for (i, j), v in zip(g["cand"].T.tolist(), p.tolist()):
            if v > 0.5 and (j not in best or v > best[j][1]):
                best[j] = (i, v)
        pairs = {(i, j) for j, (i, _) in best.items()}
    else:
        pairs = {(i, j) for (i, j), v in zip(g["cand"].T.tolist(), p.tolist()) if v > 0.5}
    return pairs


def evaluate(model, graphs, one_key):
    model.eval()
    with torch.no_grad():
        preds = [decode(g, model(g), one_key) for g in graphs]
    return link_prf(preds, [g["doc"].kv_links for g in graphs])


def run(seed, a, tr, va, tests):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = GeoLinker(tr[0]["x"].shape[1], hidden=a.hidden, layers=a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best, bad, state = -1, 0, None
    for ep in range(a.epochs):
        model.train(); random.shuffle(tr)
        for g in tr:
            if g["cand"].numel() == 0:
                continue
            s = model(g)
            pos = g["y"].sum().clamp_min(1)
            loss = F.binary_cross_entropy_with_logits(s, g["y"], pos_weight=torch.tensor(a.pos_weight))
            opt.zero_grad(); loss.backward(); opt.step()
        f = evaluate(model, va, a.one_key).f1
        if f > best:
            best, bad = f, 0; state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience:
            break
    model.load_state_dict(state)
    if a.save:
        os.makedirs("models/geolinker", exist_ok=True)
        torch.save({"state": state, "in_node": tr[0]["x"].shape[1], "args": vars(a)},
                   f"models/geolinker/{a.tag}_s{seed}.pt")
    return {name: evaluate(model, gs, a.one_key).as_dict() for name, gs in tests.items()}, best, ep + 1, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pos_weight", type=float, default=3.0)
    ap.add_argument("--one_key", action="store_true")
    ap.add_argument("--tag", default="geolinker")
    ap.add_argument("--save", action="store_true", help="save each seed's best model to models/geolinker/")
    a = ap.parse_args()
    torch.set_num_threads(4)
    train_docs = load("funsd", "train")
    rng = random.Random(0); idx = list(range(len(train_docs))); rng.shuffle(idx)
    va = [build_entity_graph(train_docs[i]) for i in idx[:15]]   # held-out val from FUNSD train
    tr = [build_entity_graph(train_docs[i]) for i in idx[15:]]
    tests = {n: [build_entity_graph(d) for d in load(n, "test")] for n in EVAL}
    out = {"args": vars(a), "runs": {}}
    t0 = time.time()
    for s in map(int, a.seeds.split(",")):
        res, vbest, eps, npar = run(s, a, list(tr), va, tests)
        out["runs"][s] = res; out["params"] = npar
        print(f"seed {s} val {vbest:.3f} epochs {eps} | " + " ".join(f"{k.replace('xfund-','')}:{v['f1']:.3f}" for k, v in res.items()) + f" | {time.time()-t0:.0f}s", flush=True)
    summ = {}
    for n in EVAL:
        v = [out["runs"][s][n]["f1"] for s in out["runs"]]
        summ[n] = {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    out["summary"] = summ
    print("MEAN " + " ".join(f"{k.replace('xfund-','')}:{v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()))
    os.makedirs("results/geolinker", exist_ok=True)
    json.dump(out, open(f"results/geolinker/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
