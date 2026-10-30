"""Extract key-value pairs from a form image with GeoLinker (no VLM).

Pipeline: docTR detects words (script-agnostic) -> words are grouped into fields ->
each field is a value if its ink is tall relative to the page's typical word (handwriting),
otherwise a key -> GeoLinker links keys to values from geometry only -> docTR's recogniser
reads the linked key and value. Recognition is the only text step and is swappable.

The role rule (relative height) is a placeholder for a trained role classifier.
"""
import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formkv.data import Document, Entity, Word  # noqa: E402
from formkv.geolinker import GeoLinker, build_entity_graph  # noqa: E402


def detect_words(path):
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True).to(dev)
    page = m(DocumentFile.from_images(path)).export()["pages"][0]
    out = []
    for b in page["blocks"]:
        for l in b["lines"]:
            for w in l["words"]:
                (x0, y0), (x1, y1) = w["geometry"]
                out.append({"text": w["value"], "conf": w["confidence"], "box": [x0, y0, x1, y1]})
    return out


def group_words(words, hand):
    """Merge words on the same row with a small gap and the same role into fields."""
    order = sorted(range(len(words)), key=lambda i: (words[i]["box"][1], words[i]["box"][0]))
    groups = []
    for i in order:
        b = words[i]["box"]; h = b[3] - b[1]
        for g in groups:
            gb = g["box"]; gh = gb[3] - gb[1]
            vov = min(b[3], gb[3]) - max(b[1], gb[1])
            gap = max(b[0] - gb[2], gb[0] - b[2])  # words of a row can arrive in any x order
            if g["hand"] == hand[i] and vov > 0.5 * min(h, gh) and -0.5 * h < gap < 1.5 * max(h, gh):
                g["idx"].append(i)
                g["box"] = [min(gb[0], b[0]), min(gb[1], b[1]), max(gb[2], b[2]), max(gb[3], b[3])]
                break
        else:
            groups.append({"idx": [i], "box": list(b), "hand": hand[i]})
    for g in groups:
        g["text"] = " ".join(words[i]["text"] for i in sorted(g["idx"], key=lambda i: words[i]["box"][0]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="models/geolinker/geolinker_onekey_s0.pt")
    ap.add_argument("--hand_ratio", type=float, default=1.42,
                    help="a word is handwriting if its height >= ratio x the page's median word height")
    ap.add_argument("--out", default="outputs/extract")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB"); W, H = img.size
    words = detect_words(a.image)
    hs = np.array([w["box"][3] - w["box"][1] for w in words])
    hand = list(hs >= a.hand_ratio * np.median(hs))
    groups = group_words(words, hand)

    px = lambda b: (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
    ents = [Entity(id=i, label="answer" if g["hand"] else "question", text=g["text"], box=px(g["box"]),
                   words=[]) for i, g in enumerate(groups)]
    doc = Document("form", "unk", W, H, [Word(g["text"], px(g["box"]), i) for i, g in enumerate(groups)],
                   ents, set(), a.image)
    g = build_entity_graph(doc)

    ck = torch.load(a.model, weights_only=False)
    model = GeoLinker(ck["in_node"], hidden=ck["args"]["hidden"], layers=ck["args"]["layers"])
    model.load_state_dict(ck["state"]); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(g)).tolist()
    best = {}
    for (i, j), v in zip(g["cand"].T.tolist(), p):  # one key per value
        if v > 0.5 and (j not in best or v > best[j][1]):
            best[j] = (i, v)

    pairs = sorted(({"key": groups[i]["text"], "value": groups[j]["text"], "score": round(v, 3),
                     "key_box": groups[i]["box"], "value_box": groups[j]["box"]}
                    for j, (i, v) in best.items()), key=lambda r: (r["value_box"][1], r["value_box"][0]))
    unlinked = [gr["text"] for j, gr in enumerate(groups) if gr["hand"] and j not in best]

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    json.dump({"pairs": pairs, "unlinked_values": unlinked}, open(f"{a.out}/{stem}.json", "w"),
              indent=1, ensure_ascii=False)
    vis = img.copy(); d = ImageDraw.Draw(vis)
    for gr in groups:
        d.rectangle(px(gr["box"]), outline=(0, 160, 0) if gr["hand"] else (120, 120, 255), width=1)
    for r in pairs:
        kb, vb = px(r["key_box"]), px(r["value_box"])
        d.rectangle(kb, outline=(0, 0, 255), width=2); d.rectangle(vb, outline=(0, 160, 0), width=2)
        d.line([((kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2), ((vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2)],
               fill=(255, 0, 0), width=2)
    vis.save(f"{a.out}/{stem}_links.png")

    for r in pairs:
        print(f"{r['score']:.2f}  {r['key'][:45]:45s} -> {r['value']}")
    print(f"unlinked values: {unlinked}")
    print(f"wrote {a.out}/{stem}.json and {stem}_links.png")


if __name__ == "__main__":
    main()
