"""Fine-tune LiLT for relation extraction on FUNSD (or an XFUND language) and evaluate on
FUNSD + all XFUND test sets (zero-shot when trained on English).

Original recipe (run_xfun_re.py, 4 GPUs): max_steps 5000, per-device batch 8 (effective 32),
lr 6.25e-6, warmup 0.1, fp16. Defaults here keep lr/warmup/steps and use effective batch 32
via gradient accumulation unless --accum is lowered for time.
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formkv.data import load, XFUND_LANGS  # noqa: E402
from formkv.lilt_re import LiltRE, chunk_document, collate  # noqa: E402
from formkv.metrics import link_prf  # noqa: E402


def evaluate(model, docs, tok, dev, bs=8):
    model.eval(); preds, chunk_golds = [], []
    with torch.no_grad():
        for d in docs:
            cs = chunk_document(d, tok); pred = set(); cg = set()
            for i in range(0, len(cs), bs):
                b = cs[i:i + bs]
                ids, bb, am = collate(b, tok.pad_token_id)
                outs = model(ids.to(dev), bb.to(dev), am.to(dev), b)
                for c, o in zip(b, outs):
                    cg |= c["chunk_gold"]
                    if o is None:
                        continue
                    for (hi, ti), y in zip(c["pairs"], o.argmax(-1).tolist()):
                        if y == 1:
                            pred.add((c["ent_ids"][hi], c["ent_ids"][ti]))
            preds.append(pred); chunk_golds.append(cg)
    return {"doc_gold": link_prf(preds, [d.kv_links for d in docs]).as_dict(),
            "chunk_gold_paper_protocol": link_prf(preds, chunk_golds).as_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/lilt-infoxlm-base")
    ap.add_argument("--tokenizer", default="models/lilt-infoxlm-base")
    ap.add_argument("--train", default="funsd")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6.25e-6)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_sets", default="funsd," + ",".join(f"xfund-{l}" for l in XFUND_LANGS))
    ap.add_argument("--out", default="results/lilt_re")
    ap.add_argument("--time_only", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoTokenizer, XLMRobertaTokenizerFast
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if "LayoutXLM" in type(tok).__name__:  # same sentencepiece vocab as xlm-roberta-base (original recipe)
        tok = XLMRobertaTokenizerFast.from_pretrained(a.tokenizer)
    model = LiltRE(a.model).to(dev)
    chunks = [c for d in load(a.train, "train") for c in chunk_document(d, tok) if c["pairs"]]
    print(f"train chunks {len(chunks)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * a.steps), a.steps)
    model.train(); t0 = time.time(); step = 0; order = []
    while step < a.steps:
        opt.zero_grad(); tot = 0.0
        for _ in range(a.accum):
            if not order:
                order = list(range(len(chunks))); random.shuffle(order)
            b = [chunks[order.pop()] for _ in range(min(a.bs, len(order)))]
            ids, bb, am = collate(b, tok.pad_token_id)
            outs = model(ids.to(dev), bb.to(dev), am.to(dev), b)
            logits = torch.cat([o for o in outs if o is not None])
            labels = torch.tensor([l for c in b for l in c["pair_labels"]], device=dev)
            loss = F.cross_entropy(logits, labels) / a.accum
            loss.backward(); tot += loss.item()
        opt.step(); sch.step(); step += 1
        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {tot:.4f} {el/step:.2f}s/step eta {(a.steps-step)*el/step/3600:.1f}h", flush=True)
        if a.time_only and step == 10:
            return
    tag = f"{a.train}_{a.model.split('/')[-1]}_s{a.seed}_steps{a.steps}_eb{a.bs*a.accum}"
    os.makedirs(a.out, exist_ok=True)
    res = {"args": vars(a), "test": {}}
    for name in a.eval_sets.split(","):
        r = evaluate(model, load(name, "test"), tok, dev)
        res["test"][name] = r
        print(f"TEST {name}: doc-gold F1 {r['doc_gold']['f1']:.4f}  paper-protocol F1 {r['chunk_gold_paper_protocol']['f1']:.4f}", flush=True)
        json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
