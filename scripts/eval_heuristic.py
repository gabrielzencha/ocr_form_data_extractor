"""Score the geometric rule baseline on FUNSD and every XFUND language (test splits)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formkv.data import iter_all
from formkv.heuristic import link_doc
from formkv.metrics import link_prf

out = {}
for name, docs in iter_all("test"):
    r = link_prf([link_doc(d) for d in docs], [d.kv_links for d in docs])
    out[name] = r.as_dict()
    print(f"{name:10s} P {r.p:.3f} R {r.r:.3f} F1 {r.f1:.3f}  (gold links {r.n_gold})")
os.makedirs("results/heuristic", exist_ok=True)
json.dump(out, open("results/heuristic/re_gold_entities.json", "w"), indent=1)
