# Handoff

Research goal: text-free, language-agnostic key-value extraction from form images.
Background and feasibility: [literature/LITERATURE_REVIEW.md](literature/LITERATURE_REVIEW.md).
All numbers below are F1 for question-to-answer linking with gold entities and gold roles
(the standard LayoutXLM/LiLT relation-extraction protocol), trained on FUNSD train only.

## Setup on a new machine

```bash
conda create -n formkv python=3.11 -y && conda activate formkv
pip install torch torchvision transformers datasets accelerate seqeval scipy numpy pandas \
  pillow tqdm pyyaml scikit-learn networkx pulp huggingface_hub opencv-python-headless matplotlib pytest
bash scripts/download_data.sh funsd xfund          # -> data/raw/
mkdir -p third_party && cd third_party
for r in Bhanu068/LAGNN ZeningLin/PEneo herobd/FUDGE jpWang/LiLT; do git clone --depth 1 https://github.com/$r.git; done
cd .. && mkdir -p models/lilt-infoxlm-base
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors; do
  curl -sSL -o models/lilt-infoxlm-base/$f https://huggingface.co/SCUT-DLVCLab/lilt-infoxlm-base/resolve/main/$f; done
```

The Hugging Face Xet downloader stalled on the Mac, so the checkpoint is fetched with curl.

## Code map

| File | What it does |
|---|---|
| `src/formkv/data.py` | One schema for FUNSD and XFUND: words, entities, directed links |
| `src/formkv/metrics.py` | Standard link F1, pair-extraction F1, LAGNN's edge macro-F1 |
| `src/formkv/heuristic.py` | Text-free rule: nearest question to the left or above |
| `src/formkv/egat.py` | Pure-PyTorch port of LAGNN's EGAT (no DGL, no Gurobi) |
| `src/formkv/graph.py` | Gold-free k-nearest-neighbour word graphs |
| `src/formkv/geolinker.py` | Text-free entity-level linker; scores every question-answer pair |
| `src/formkv/lilt_re.py` | LiLT encoder plus the official relation-extraction head |
| `scripts/replicate_lagnn.py`, `eval_lagnn_entity_level.py` | LAGNN replication and rescoring |
| `scripts/eval_heuristic.py`, `train_geolinker.py`, `train_lilt_re.py` | Baselines and the proposed linker |

## Results so far

| Method | FUNSD | XFUND zero-shot, 7 languages |
|---|---|---|
| LiLT, published | 0.628 | 0.42–0.51 |
| LayoutXLM-large, published | — | 0.48–0.57 |
| GeoLayoutLM, published | 0.895 | — |
| Geometric rule, no text | 0.874 | 0.73–0.79 |
| GeoLinker, no text, one key per value, 2 of 5 seeds (partial) | 0.939–0.946 | 0.81–0.94 |

Final GeoLinker summaries land in `results/geolinker/*.json` (see `summary`).

## Findings to keep

- **LAGNN replicated.** With the paper's seed 6222, the port matches all 7 reported numbers within 0.016 (English 0.844 vs 0.845). Seeds 1 and 2 give about 0.80 on English. See `results/lagnn_replication/`.
- **LAGNN's released test graphs leak gold structure.** Every gold relation is inserted as a candidate edge, and negatives are sampled with different geometry. A gradient-boosted classifier separates no-link edges with 1 error in 51,775 FUNSD test edges, and at most 58 errors in 176k on XFUND. Rescored at entity level, its precision is exactly 1.0. Its numbers measure relation typing, not relation discovery.
- **The released LAGNN code crashes on its own files.** It does not map the `word-word` label, which we map to same-entity.
- **The gold-roles protocol is nearly saturated by geometry.** The real difficulty is role classification and entity grouping, which is where the paper should aim.
- **Word-level kNN graphs miss gold links.** XFUND Chinese and Japanese words are single characters. Use segment or entity level.

## Next steps

1. **Make the LiLT trainer use CUDA.** `scripts/train_lilt_re.py` picks MPS or CPU only; add `cuda` first. Then run the original recipe (5,000 steps, effective batch 32, lr 6.25e-6) and check it reproduces 0.628 on FUNSD. On the Mac it was too slow to measure.
2. **Run the end-to-end protocol with no gold entities or roles.** Predict roles and grouping too, and score with pair-extraction F1 on RFUND (from PEneo) against LiLT, LayoutXLM and PEneo.
3. **Add visual features to close the role gap.** Crop CNN features, FFDNet slot detections, and the form-dropout self-supervision idea from the review.
4. **Build the multi-script benchmark.** Use BaFCo (Bangla) plus synthetic fills of CommonForms blank pages in low-resource scripts.
