"""Unified document schema and loaders for FUNSD and XFUND.

Every dataset is converted to `Document`: words with boxes, entities that group
words and carry a role label, and directed links. Boxes are kept in pixels; use
`Document.norm_box` for [0, 1] coordinates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image

ROLE_LABELS = ("header", "question", "answer", "other")


@dataclass
class Word:
    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    entity: int  # index into Document.entities


@dataclass
class Entity:
    id: int
    label: str  # one of ROLE_LABELS
    text: str
    box: tuple[float, float, float, float]
    words: list[int] = field(default_factory=list)  # indices into Document.words


@dataclass
class Document:
    doc_id: str
    lang: str
    width: int
    height: int
    words: list[Word]
    entities: list[Entity]
    links: set[tuple[int, int]]  # (from_entity_idx, to_entity_idx), deduplicated, directed
    image_path: str | None = None

    def norm_box(self, box):
        x0, y0, x1, y1 = box
        return (x0 / self.width, y0 / self.height, x1 / self.width, y1 / self.height)

    @property
    def kv_links(self) -> set[tuple[int, int]]:
        """question -> answer links: the standard RE target on FUNSD/XFUND."""
        return {
            (a, b) for a, b in self.links
            if self.entities[a].label == "question" and self.entities[b].label == "answer"
        }


def _orient(a: Entity, b: Entity) -> tuple[int, int] | None:
    """Orient an undirected annotation link as (parent, child)."""
    order = {"header": 0, "question": 1, "answer": 2, "other": 3}
    la, lb = order.get(a.label, 3), order.get(b.label, 3)
    if la < lb:
        return a.id, b.id
    if lb < la:
        return b.id, a.id
    return None  # same-role links (e.g. question-question) carry no direction


def _build(doc_id, lang, width, height, form, image_path) -> Document:
    words: list[Word] = []
    entities: list[Entity] = []
    id2idx: dict[int, int] = {}
    for item in form:
        text = item.get("text", "")
        ws = [w for w in item.get("words", []) if w.get("text", "").strip() != ""]
        if not ws:
            continue  # FUNSD has a handful of empty entities; dropping them matches LayoutLM preprocessing
        e_idx = len(entities)
        id2idx[item["id"]] = e_idx
        ent = Entity(id=e_idx, label=item["label"], text=text, box=tuple(map(float, item["box"])))
        for w in ws:
            ent.words.append(len(words))
            words.append(Word(text=w["text"], box=tuple(map(float, w["box"])), entity=e_idx))
        entities.append(ent)
    links: set[tuple[int, int]] = set()
    for item in form:
        for a, b in item.get("linking", []):
            if a in id2idx and b in id2idx and a != b:
                o = _orient(entities[id2idx[a]], entities[id2idx[b]])
                if o is not None:
                    links.add(o)
    return Document(doc_id, lang, width, height, words, entities, links, image_path)


def load_funsd(root: str, split: str) -> list[Document]:
    """split: 'train' (149 docs) or 'test' (50 docs)."""
    sub = {"train": "training_data", "test": "testing_data"}[split]
    ann_dir = os.path.join(root, "dataset", sub, "annotations")
    img_dir = os.path.join(root, "dataset", sub, "images")
    docs = []
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        img_path = os.path.join(img_dir, stem + ".png")
        with Image.open(img_path) as im:
            w, h = im.size
        with open(os.path.join(ann_dir, fn)) as f:
            form = json.load(f)["form"]
        docs.append(_build(stem, "en", w, h, form, img_path))
    return docs


def load_xfund(root: str, lang: str, split: str) -> list[Document]:
    """split: 'train' (149 docs) or 'val' (50 docs; the XFUND test set)."""
    with open(os.path.join(root, f"{lang}.{split}.json")) as f:
        data = json.load(f)
    docs = []
    for d in data["documents"]:
        img = d["img"]
        img_path = os.path.join(root, f"{lang}.{split}", img["fname"])
        docs.append(_build(d["id"], lang, img["width"], img["height"], d["document"], img_path))
    return docs


XFUND_LANGS = ("zh", "ja", "es", "fr", "it", "de", "pt")


def load(dataset: str, split: str, data_root: str = "data/raw") -> list[Document]:
    """dataset: 'funsd' or 'xfund-<lang>'. split: 'train' or 'test'."""
    if dataset == "funsd":
        return load_funsd(os.path.join(data_root, "funsd"), split)
    if dataset.startswith("xfund-"):
        lang = dataset.split("-", 1)[1]
        return load_xfund(os.path.join(data_root, "xfund"), lang, "val" if split == "test" else split)
    raise ValueError(f"unknown dataset {dataset}")


def iter_all(split: str, data_root: str = "data/raw") -> Iterator[tuple[str, list[Document]]]:
    yield "funsd", load("funsd", split, data_root)
    for lang in XFUND_LANGS:
        yield f"xfund-{lang}", load(f"xfund-{lang}", split, data_root)
