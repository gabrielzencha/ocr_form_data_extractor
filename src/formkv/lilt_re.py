"""LiLT + REDecoder relation extraction, following jpWang/LiLT examples/run_xfun_re.py.

Preprocessing matches LiLT's XFUND loader: each entity's text is tokenised as a unit, all
its tokens get the entity box (0-1000), documents are cut into 512-token chunks, and only
entities fully inside a chunk (and relations between them) are kept. Entity labels for the
decoder: header=0, question=1, answer=2; 'other' entities are excluded, as in the original.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .data import Document

ENT_LABEL = {"header": 0, "question": 1, "answer": 2}


class BiaffineAttention(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.bilinear = nn.Bilinear(d, d, out, bias=False)
        self.linear = nn.Linear(2 * d, out, bias=True)

    def forward(self, a, b):
        return self.bilinear(a, b) + self.linear(torch.cat((a, b), dim=-1))


class REDecoder(nn.Module):
    """Copy of LiLTfinetune/modules/decoders/re.py, returning per-pair logits."""

    def __init__(self, hidden, dropout):
        super().__init__()
        self.entity_emb = nn.Embedding(3, hidden, scale_grad_by_freq=True)
        proj = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout))
        self.ffnn_head, self.ffnn_tail = copy.deepcopy(proj), copy.deepcopy(proj)
        self.rel_classifier = BiaffineAttention(hidden // 2, 2)

    def forward(self, hs, starts, labels, heads, tails):
        h = torch.cat((hs[starts[heads]], self.entity_emb(labels[heads])), -1)
        t = torch.cat((hs[starts[tails]], self.entity_emb(labels[tails])), -1)
        return self.rel_classifier(self.ffnn_head(h), self.ffnn_tail(t))


class LiltRE(nn.Module):
    def __init__(self, name="SCUT-DLVCLab/lilt-infoxlm-base"):
        super().__init__()
        from transformers import LiltModel
        self.enc = LiltModel.from_pretrained(name)
        cfg = self.enc.config
        self.dec = REDecoder(cfg.hidden_size, cfg.hidden_dropout_prob)

    def forward(self, input_ids, bbox, attention_mask, chunks):
        hs = self.enc(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask).last_hidden_state
        out = []
        for b, c in enumerate(chunks):
            if not c["pairs"]:
                out.append(None); continue
            dev = hs.device
            heads = torch.tensor([p[0] for p in c["pairs"]], device=dev)
            tails = torch.tensor([p[1] for p in c["pairs"]], device=dev)
            out.append(self.dec(hs[b], torch.tensor(c["starts"], device=dev),
                                torch.tensor(c["labels"], device=dev), heads, tails))
        return out


def chunk_document(doc: Document, tok, max_len: int = 512) -> list[dict]:
    """Tokenise a document LiLT-style and split into chunks.

    Returns chunks with: input_ids, bbox, starts (token idx of each kept entity), labels,
    ent_ids (doc entity index), pairs (all question->answer candidate pairs, local idx),
    pair_labels (1 if gold link).
    """
    ids, boxes, spans = [], [], []
    for e in doc.entities:
        t = tok(e.text if e.text.strip() else " ".join(doc.words[w].text for w in e.words),
                add_special_tokens=False)["input_ids"]
        if not t:
            continue
        x0, y0, x1, y1 = doc.norm_box(e.box)
        bb = [int(min(max(v, 0), 1) * 1000) for v in (x0, y0, x1, y1)]
        spans.append((e.id, len(ids), len(ids) + len(t)))
        ids += t; boxes += [bb] * len(t)
    body = max_len - 2
    chunks = []
    gold = doc.kv_links
    for s in range(0, max(len(ids), 1), body):
        e_ = s + body
        ents = [(eid, a - s + 1) for eid, a, b in spans
                if a >= s and b <= e_ and doc.entities[eid].label in ENT_LABEL]
        loc = {eid: i for i, (eid, _) in enumerate(ents)}
        labels = [ENT_LABEL[doc.entities[eid].label] for eid, _ in ents]
        pairs, plab = [], []
        for i, (ei, _) in enumerate(ents):
            if labels[i] != 1:
                continue
            for j, (ej, _) in enumerate(ents):
                if labels[j] == 2:
                    pairs.append((i, j)); plab.append(int((ei, ej) in gold))
        chunks.append({
            "input_ids": [tok.cls_token_id] + ids[s:e_] + [tok.sep_token_id],
            "bbox": [[0, 0, 0, 0]] + boxes[s:e_] + [[1000, 1000, 1000, 1000]],
            "starts": [st for _, st in ents], "labels": labels, "ent_ids": [eid for eid, _ in ents],
            "pairs": pairs, "pair_labels": plab,
            "chunk_gold": {(ents[i][0], ents[j][0]) for (i, j), l in zip(pairs, plab) if l},
        })
    return chunks


def collate(chunks, pad_id):
    L = max(len(c["input_ids"]) for c in chunks)
    ids = torch.full((len(chunks), L), pad_id, dtype=torch.long)
    bb = torch.zeros((len(chunks), L, 4), dtype=torch.long)
    am = torch.zeros((len(chunks), L), dtype=torch.long)
    for i, c in enumerate(chunks):
        n = len(c["input_ids"])
        ids[i, :n] = torch.tensor(c["input_ids"]); bb[i, :n] = torch.tensor(c["bbox"]); am[i, :n] = 1
    return ids, bb, am
