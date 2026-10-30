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

On Linux, pip's default torch wheels target CUDA 13 and need NVIDIA driver 580 or newer. With an older driver (the L40 box has 550), install torch first with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

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
| LiLT, published | 0.628 | 0.42–0.51 (avg 0.493) |
| LiLT, our port, original recipe, seed 42 | 0.625 | 0.40–0.50 (avg 0.447) |
| LiLT, our port, decoder lr x40, seed 42 | 0.699 | 0.52–0.64 (avg 0.562) |
| LayoutXLM-large, published | — | 0.48–0.57 |
| GeoLayoutLM, published | 0.895 | — |
| KVPFormer, published, gold labels (current SOTA we know of) | 0.909 (0.956 for EN in its XFUND table) | 0.86–0.94 (avg 0.897) |
| Geometric rule, no text | 0.874 | 0.73–0.79 |
| GeoLinker, no text, all pairs, 5 seeds | 0.895 ± 0.019 | 0.77–0.86 |
| GeoLinker, no text, one key per value, 5 seeds | 0.944 ± 0.005 | 0.83–0.93 |

Final GeoLinker summaries land in `results/geolinker/*.json` (see `summary`).
GeoLinker reproduces across machines: re-run on the Linux/L40 box (CPU, torch 2.14), the 5-seed means are FUNSD 0.890 ± 0.014 (all pairs) and 0.942 ± 0.003 (one key), and all 16 means are within 0.015 of the Mac results (`*_rerun_l40.json`).

## Findings to keep

- **LAGNN replicated.** With the paper's seed 6222, the port matches all 7 reported numbers within 0.016 (English 0.844 vs 0.845). Seeds 1 and 2 give about 0.80 on English. See `results/lagnn_replication/`.
- **LAGNN's released test graphs leak gold structure.** Every gold relation is inserted as a candidate edge, and negatives are sampled with different geometry. A gradient-boosted classifier separates no-link edges with 1 error in 51,775 FUNSD test edges, and at most 58 errors in 176k on XFUND. Rescored at entity level, its precision is exactly 1.0. Its numbers measure relation typing, not relation discovery.
- **The released LAGNN code crashes on its own files.** It does not map the `word-word` label, which we map to same-entity.
- **The gold-roles protocol is nearly saturated by geometry.** The real difficulty is role classification and entity grouping, which is where the paper should aim.
- **Word-level kNN graphs miss gold links.** XFUND Chinese and Japanese words are single characters. Use segment or entity level.
- **LiLT reproduced on FUNSD (0.625 vs 0.628), slightly low on XFUND zero-shot (0.447 vs 0.493 avg).** Single seed, paper protocol (gold links whose entities fall in one 512-token chunk); counting every document link gives 0.621 and 0.411. Per language: zh 0.456, ja 0.456, es 0.476, fr 0.496, it 0.440, de 0.403, pt 0.400. See `results/lilt_re/`.
- **The first port of LiLT's RE head got 0.304 on FUNSD; five mismatches with the official code fixed it.** Hugging Face's `LiltModel` drops the layout stream, but the official decoder reads text + layout concatenated (768 + 192), so we recover it with a forward hook. The decoder also needs normal(0, 0.02) init on its Linear/Embedding layers, dropout on its input, 4x learning rate on its layers except `rel_classifier` (`speedup_r` in `xfun_trainer.py`, absent from the README), and weight decay 0. Without them the loss sits at the class-prior entropy (about 0.17) for the whole run. Even with them, the loss only leaves that plateau around step 1,250, so short runs look broken.
- **LiLT's published RE numbers understate it; a faster decoder gives a stronger baseline.** With `--head_lr_mult 40` (decoder lr 2.5e-4, everything else unchanged) the loss leaves the plateau by step 500, and the scores reach FUNSD 0.699 and XFUND zero-shot avg 0.562 (zh 0.583, ja 0.518, es 0.584, fr 0.641, it 0.570, de 0.516, pt 0.524). Training loss reaches 0.003, so it fits the train set, yet test scores still rise. Compare GeoLinker against this stronger LiLT too, not only the published numbers. See `results/lilt_re_headlr40/`.

## Next steps

1. **Run more LiLT seeds, for both recipes.** The original recipe reproduces on FUNSD with one seed (42), and lr x40 beats it with one seed. 2 to 4 more seeds of each would show whether the XFUND gap of about 0.05 below the paper, and the lr x40 gain, are larger than seed noise. Each run takes about 45 minutes on an L40 (about 1.5 hours with two in parallel). A short sweep of `--head_lr_mult` (10, 40, 100) would also help.
2. **Run the end-to-end protocol with no gold entities or roles.** Predict roles and grouping too, and score with pair-extraction F1 on RFUND (from PEneo) against LiLT, LayoutXLM and PEneo.
3. **Add visual features to close the role gap.** Crop CNN features, FFDNet slot detections, and the form-dropout self-supervision idea from the review.
4. **Build the multi-script benchmark.** Use BaFCo (Bangla) plus synthetic fills of CommonForms blank pages in low-resource scripts.
