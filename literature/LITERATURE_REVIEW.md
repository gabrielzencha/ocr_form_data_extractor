# Literature Review — Language-Agnostic, Low-LM Form Data Extraction from Document Images

**Compiled:** 2026-10-29
**Scope:** Extracting key→value data from form images by exploiting the visual/geometric structure of forms ("Name: ______", "Region: [ ]", boxed answers), with (a) no dependence on a large language model or VLM, (b) no per-template retraining, and (c) portability to scripts/languages the model has never seen.
**Papers:** ~85 PDFs in [papers/](papers/) (organised by theme; see [papers/INDEX.md](papers/INDEX.md)). Files are named `<arxivId>_<slug>.pdf`.

---

## 0. Verdict in one paragraph

Yes, this is doable, and three independent groups have already shown the core claim empirically: **key–value *linking* in forms is overwhelmingly a geometric/visual problem, and text semantics are only needed for a minority of cases.** The closest prior work is **LAGNN** (Voutharoja, Qu & Shiri, ICDAR 2023, [2305.04460](papers/03_visual_language_agnostic/2305.04460_lagnn_language_independent_neurosymbolic.pdf)): a graph network with ~8K parameters that uses only bounding-box geometry (no transcriptions, no language model), beats LayoutLMv3 on FUNSD relation extraction under its word-graph metric (0.854 vs 0.805 F1) and, trained on 139 English forms, zero-shots to Japanese/Chinese XFUND at roughly double the score of LiLT/LayoutXLM. **FUDGE** (Davis et al., ICDAR 2021) and **GeoContrastNet** (ICDAR 2024) reach the same qualitative conclusion with purely visual features. What *nobody* has done is turn this into a complete, query-able extraction system that (i) also detects empty value slots/checkboxes, (ii) is evaluated on low-resource scripts rather than XFUND's seven high-resource languages, (iii) exposes a query interface (by string, by example image, by schema), and (iv) treats OCR as a swappable, last-stage component. That combination is a credible research contribution, and the risks are concentrated in one place: **key-vs-value role classification without text**, where visual-only methods lose 10–20 F1 points to text-based ones. Sections 3–4 give the evidence and a concrete plan.

---

## 1. Problem definition and decomposition

The task the proposal describes is *template-free, language-agnostic key-value extraction* (KVE). Across the literature it is decomposed into the following sub-problems, which is also the natural module structure for a system:

| # | Sub-problem | Language-dependence | Best evidence |
|---|---|---|---|
| A | **Text / field-element detection**: text lines, words, blank slots ("____", boxes), checkboxes, signature fields | Essentially none (detection is script-agnostic in practice) | CommonForms (2509.16506), AutoFormBench (2603.29832), DB/CRAFT detectors in PaddleOCR/docTR |
| B | **Role classification**: is a text block a *key* (pre-printed prompt), a *value* (filled answer), a header, or other? | Medium. Text helps; visual cues (print vs. handwriting, font weight, position relative to slots) partly substitute | FUDGE, LAGNN, Revising-FUNSD (2010.05322) |
| C | **Key–value linking** (entity linking / relation extraction) | Low. Geometry dominates | LAGNN, FUDGE, GeoLayoutLM, PEneo |
| D | **Grouping / hierarchy**: multi-line entities, sections, repeated groups, tables | Low–medium | PEneo (line grouping), DocStruct, Form2Seq, DocILE line items |
| E | **Value transcription** (OCR of the answer) | High, but *swappable*: train/choose a recogniser per script | Tesseract, PaddleOCR 3.0, Surya; Sinhala/Tamil study (2507.18264) |
| F | **Query interface**: "give me the value of *this* prompt" by string, by example crop, or by schema | Depends on the query type | QueryForm (schema), one-shot graph matching (example), word spotting / PHOC (example image) |

The key architectural insight supported by the literature (§3) is that **A–D can be solved from boxes + pixels alone**, and E can be deferred to the end and applied only to the value regions you actually need. That is the opposite of VLM / Donut-style systems, which fuse recognition and structure into one decoder whose vocabulary and training data fix the languages it can read.

---

## 2. Review by approach family

### 2.1 Classical template-based forms processing (1988–2005)

The original "forms processing" literature already treated the problem as a *visual* one: register the scanned form to a known template, remove the pre-printed layer ("form dropout"), and read what remains in known field locations.

- Casey, Ferguson, Mohiuddin & Walach, *Intelligent forms processing system*, Machine Vision & Applications, 1992. Form identification, dropout, ICR, contextual checking.
- Taylor, Fritzson & Pastor, *Extraction of data from preprinted forms*, Machine Vision & Applications, 1992.
- Yu & Jain, *A generic system for form dropout*, IEEE TPAMI, 1996. Separates pre-printed lines/boxes from filled-in data using morphology.
- Cesarini, Gori, Marinai & Soda, *INFORMys: a flexible invoice-like form-reader system*, IEEE TPAMI, 1998. Attributed relational graphs + hypothesise-and-verify for field location.
- Lam, Javanbakht & Srihari, *Anatomy of a form reader*, ICDAR 1993.

(These are not on arXiv and sit behind Springer/IEEE paywalls, so they were not downloaded; CMU library access will get all of them.)

**What to keep:** the *pre-printed vs. filled* separation is a language-independent visual signal that modern deep KVE largely ignores, and it is exactly the cue behind "Name: ______". **What to drop:** per-template registration, which is the brittleness the proposal wants to avoid. The modern "template-free" line (§2.5) starts explicitly as a reaction to this.

### 2.2 Grid / pixel-level document representations

A second lineage renders the document into a 2-D grid and applies fully-convolutional segmentation, which makes KVE a vision problem.

- **Chargrid** (Katti et al., EMNLP 2018, 1809.08799) — one-hot character identities painted into a grid; encoder–decoder predicts field segmentation + boxes. Beats sequence-text and raw-image baselines on invoices. Character one-hots are script-specific, but the *formulation* is not.
- **Chargrid-OCR** (1909.04469) — end-to-end OCR as instance segmentation in the same framework.
- **BERTgrid** (1909.04948), **ViBERTgrid** (ICDAR 2021, 2105.11672) — replace characters with contextual embeddings; stronger but language-bound.
- **CUTIE** (1903.12363), **VisualWordGrid** (2010.02358) — grid variants that add image features; VisualWordGrid shows that image channels alone carry useful signal.
- **Revising FUNSD for key-value detection** (2010.05322) — the most relevant grid paper: a U-Net takes the *text mask + grayscale image* (no text content) and segments key / value / other pixels on FUNSD. Reported mean IoU 0.79–0.80 (4-class) and an ablation showing text-mask-only 0.55 → text-mask+image 0.69 mIoU, i.e. visual appearance (not just layout) is informative for the key/value role. The authors also re-annotated FUNSD and documented its label inconsistencies.

**Takeaway:** a "box-grid" or "mask-grid" representation with no character identities is a legitimate language-agnostic input and is a cheap way to inject visual role cues (underlines, boxes, print vs. handwriting) into a segmentation-style key/value classifier.

### 2.3 Layout-aware pre-trained language models (the mainstream, and why it is language-bound)

Since 2020 the dominant paradigm is a transformer over OCR tokens + 2-D positions (+ image patches), pre-trained on millions of documents:

LayoutLM (1912.13318), LayoutLMv2 (2012.14740), LayoutLMv3 (2204.08387), BROS (2108.04539), DocFormer (2106.11539) / DocFormerv2 (2306.01733), StructuralLM (2105.11210), XYLayoutLM (2203.06947), LAMBERT (2002.08087), TILT (2102.09550), UDOP (2212.02623), ERNIE-Layout (2210.06155), StrucTexT (2108.02923), GraphDoc (2203.13530), FormNet (2203.08411) and FormNetV2 (2305.02549). Surveys: Cui et al. (2111.08609), Ding et al. (2408.01287).

All of these inherit a tokenizer and a text encoder. Two papers address languages directly:

- **LayoutXLM / XFUND** (2104.08836): pre-trained on 30M documents in 53 languages; XFUND adds forms in ZH/JA/ES/FR/IT/DE/PT (199 forms per language). Relation-extraction F1 (large model): language-specific avg 0.72, **zero-shot from English avg 0.55**, multitask 0.85.
- **LiLT** (Wang, Jin & Ding, ACL 2022, 2202.13669): decouples the layout stream from the text stream (BiACM), so a layout model pre-trained on *English only* (11M docs) can be fine-tuned with *any* text encoder. LiLT[InfoXLM] beats LayoutXLM on every XFUND language (RE avg 0.678 vs 0.643 language-specific; 0.493 vs 0.439 zero-shot). This is the strongest statement in the mainstream literature that **layout knowledge transfers across languages while text knowledge does not**, and it is the natural "text-optional" baseline for the proposal.

**Limitation that matters here:** XFUND covers only seven high-resource languages in Latin/CJK scripts; LiLT's language coverage is still bounded by the chosen text encoder's tokenizer. There is no evidence in this family for scripts outside XLM-R's training set.

### 2.4 Graph- and geometry-centric key–value linking

A parallel line treats the document as a graph of text segments and learns the links; it is where the best linking numbers come from.

- **Liu et al., GCN for VRDs** (1903.11279) and **PICK** (2004.07464): early graph-conv KIE with text + geometry.
- **SPADE** (Hwang et al., Findings ACL 2021, 2005.00642): KIE as *spatial dependency parsing* — predict a relation matrix among tokens rather than IOB tags, which handles nested/hierarchical fields.
- **DocStruct** (Findings EMNLP 2020, 2010.11685): forms as a hierarchy; key–value as parent–child links; multimodal fusion.
- **Doc2Graph** (ECCV-W 2022, 2208.11168): small task-agnostic GNN (6.2M params) for entity linking, invoice layout and table detection; FUNSD EL F1 0.53.
- **GeoLayoutLM** (CVPR 2023, 2304.10759): adds *geometric* pre-training tasks and relation heads; FUNSD RE F1 80.4 → **89.5** (entity-level metric). The improvement is attributed entirely to explicit geometry modelling.
- **GOSE** (2305.13850): global-structure-guided iterative relation refinement.
- **KVPFormer** (AAAI 2023, 2304.07957): keys as "questions", a decoder predicts values in parallel, with a *spatial compatibility attention bias*; +7.2 / +13.2 F1 over prior SOTA on FUNSD / XFUND.
- **UniVIE** (2401.09220): unified label space for KVE; **XFormParser** (2405.17336): LayoutXLM backbone with a joint SER+RE head and InDFormBench.
- **PEneo / RFUND** (ACM MM 2024, 2401.03472): the most important *engineering* result for the proposal. (i) FUNSD/XFUND annotation granularity is inconsistent, so they release **RFUND**. (ii) A pipeline of entity recognition → relation extraction accumulates error (a 19-pt SER drop ⇒ 15-pt pair-extraction drop). (iii) Solving *line extraction + line grouping + linking jointly* on a token-pair representation raises pair-extraction F1 by **≈20 points** on RFUND-EN for *every* backbone (e.g. LayoutLMv3 57.7 → 79.3; LiLT 52.2 → 74.3) with no extra pre-training. The decoder design, not the backbone, is the lever.
- **P²Net / CLEX** (Findings ACL 2025; preprint PPN 2307.10551): pointer-based KIE where keys and values point at each other; CLEX dataset (5,860 images, 1,162 entity categories) for zero/few-shot KIE.
- **See-then-Tell** (2409.19573), **GraphRevisedIE** (2410.01160): recent grounding/graph refinements.
- **Identify, Locate, Link** (IBM, 2608.20868): a 256M-parameter SmolDocling VLM fine-tuned to emit key/value/link tags; beats zero-shot 7B VLMs on FUNSD/XFUND under a layout-aware metric. Included as the current "small end-to-end" reference point; note it still has a script-bound decoder.

**Takeaway:** geometry-aware linking heads consistently win, and joint grouping+linking decoding is worth ~20 F1 regardless of backbone. Both ideas transfer directly to a text-free backbone.

### 2.5 Visual-only and language-independent form understanding — the closest prior work

These are the papers that have "come close" to the proposal.

| Paper | Inputs | FUNSD linking | FUNSD entity/role | Cross-lingual | Notes |
|---|---|---|---|---|---|
| **Davis et al. 2019**, *Deep Visual Template-Free Form Parsing* (ICDAR 2019, 1909.02576) | pixels only | — (NAF: 60.7 F1 key–value, simple subset) | detects pre-printed vs handwritten | not tested | Introduces **NAF** (historical handwritten forms, no transcriptions); learned pairing beats heuristic rules; language-agnostic by construction |
| **FUDGE** (ICDAR 2021, 2105.08194) | pixels + geometry (no text); "Word-FUDGE" adds OCR *boxes* only | **56.6** F1 (no text), 62.6 with word boxes; vs BROS 67.0 (138M params, 10M-doc pre-training) | 66.5 (vs 78–84 for LM-based) | not tested (motivated by resource-poor languages) | 17M params, trained on FUNSD only. Most false negatives come from entity detection, not linking. GCN adds little |
| **LAGNN** (ICDAR 2023, 2305.04460) | OCR *boxes* only (docTR), 6-D relative-spacing edge features; 1-layer GAT, 3 heads; rules + ILP | **0.854** word-graph RE F1 (LayoutLMv3 0.805, LiLT 0.789, LayoutXLM 0.712) | 0.928 on its entity graph | **Zero-shot XFUND avg 0.669** vs LiLT 0.553 / LayoutXLM 0.464; JA 0.625 vs 0.303; ZH 0.626 vs 0.349 | ~8K parameters, trained on 139 forms. ⚠ Metric is word-pair-level on their own relation labels, not the standard entity-level RE used for GeoLayoutLM's 89.5 — compare with care |
| **GeoContrastNet** (ICDAR 2024, 2405.03104) | box geometry (+ MobileNet crops), contrastive edge learning, GAT | 0.32 EL (GT boxes); over-predicts links | 0.65 SER | claimed, untested | Weak numbers but a clean ablation: geometry-only > visual-only, combined best |
| **Revising FUNSD** (2010.05322) | text mask + grayscale | — (segmentation) | mIoU 0.69–0.80 | — | Visual appearance helps role classification |
| Vision-only pre-training: **DiT** (2203.02378), **StrucTexTv2** (ICLR 2023, 2303.00289) | image only at inference | — | — | pre-training text is EN/ZH | Useful image encoders; StrucTexTv2's masked *textual* prediction is still language-specific |

**Reading of the evidence.** (1) Linking: a geometry-only model (LAGNN) is *competitive with or better than* text-based models on English forms and far better zero-shot on other languages. (2) Role classification is where language helps: FUDGE and GeoContrastNet are 10–20 points behind LM-based models on entity detection/SER, while LAGNN's 0.928 is on an easier, graph-derived formulation. (3) None of these works evaluates on a script absent from XFUND, on blank forms, or with a query interface; GeoContrastNet and FUDGE both state multilingual evaluation as *future work*. These are precisely the open slots.

### 2.6 Form structure and field-element detection (blank forms, widgets)

The proposal's "Name: ______" intuition needs a detector for *value slots*, including empty ones. This has its own literature, mostly from Adobe and recent dataset work:

- **Multi-modal association based grouping for form structure extraction** (Adobe, WACV 2020, 2107.04396) and **Form2Seq** (EMNLP 2020, 2107.04419): classify low-level elements (text runs, widgets, choice fields) and group them into TextFields / ChoiceFields / ChoiceGroups; F1 75.8 / 86.0 / 61.6.
- **CommonForms / FFDNet** (WACV 2026, 2509.16506): 480k pages of real PDF forms from Common Crawl (≈36% non-English), three classes (text input, choice button, signature); YOLO11 detectors trained from scratch at 1216px reach mAP50-95 **81.0** (FFDNet-L, 25M params, <$500 to train). Pure detection, language-agnostic, but *no semantic grouping of fields to their prompts* — that is the missing step your work would add.
- **AutoFormBench** (2603.29832): 407 real forms; YOLOv11 vs OpenCV contour detection for checkboxes / input lines / text boxes.

**Takeaway:** empty-slot and widget detection is a solved-enough, script-agnostic component with open weights; linking slots to prompts is not.

### 2.7 Zero-shot, few-shot, one-shot and query-driven extraction

- **QueryForm** (Google, Findings ACL 2023, 2211.07730): schema prompt + entity prompt → text+layout transformer (no image); pre-trained on 13M weakly-labelled multilingual web pages. Zero-shot XFUND avg 69.3 vs LayoutXLM 55.6–61.2. Strong, but the query is *textual* and the model inherits a multilingual text encoder.
- **Field Extraction from Forms with Unlabeled Data** (Salesforce, SpaNLP 2022, 2110.04282): pseudo-labels from *simple rules* — fuzzy string match of a key list + a geometric score rewarding candidates close to, below, or right of the key — then a LayoutLM refined by progressive label ensembling. Rules alone reach 44 F1 on invoices; training on rule labels 61–64. Shows how far hand-written geometric priors get and how to bootstrap from them.
- **One-shot KIE via deep partial graph matching** (2109.13967) and **LF-BP** (ACM MM 2020, 2009.04153): one annotated *support* form labels a *query* form by matching field graphs (spatial + aspect + word-embedding features). Accuracy 95–97% on same-template documents, including drifted fields; assumes one support doc per layout.
- **Attend, Copy, Parse** (1812.07248): early end-to-end extraction with copy mechanisms.
- Esser et al., *Few-exemplar information extraction for business documents* (ICEIS 2014): 78 F1 with one example per template.

**Takeaway:** "query by example" against a support document exists but is template-bound and text-dependent; "query by string/schema" exists but is LM-dependent. A geometry/visual-first system with a query layer on top is not in the literature.

### 2.8 OCR-free end-to-end models and VLMs — the approach being avoided, and the evidence that supports avoiding it

- **Donut** (ECCV 2022, 2111.15664), **Pix2Struct** (2210.03347), **Dessurt** (2203.16618): image → sequence decoders. Schema changes require retraining (the proposal's objection), and the decoder tokenizer fixes the readable scripts.
- **Survey of MLLM-based VRDU** (Ding et al., 2507.09861; Findings ACL 2026) — documents the shift to OCR-free MLLMs and lists domain adaptation and synthetic-data quality as open problems.
- **GlotOCR Bench** (2604.12978): most OCR/VLM systems perform well on <10 scripts; frontier models fail beyond ~30; one frontier model drops from 95.3% (high-resource) to **7.7%** on low-resource scripts, and models produce fluent output in the *wrong* script rather than signalling uncertainty.
- **ThaiOCRBench** (2511.04479): language bias, structural mismatch and hallucination as dominant VLM error types.
- **BaFCo** (2607.05614): 200 Bangla government forms, 1,926 KV pairs; MLLMs "have limitations ... particularly in accurately localizing highly granular form entities".

**Takeaway:** the proposal's premise (VLMs fail silently on unsupported scripts; Donut needs retraining) is well documented as of 2026.

### 2.9 Multilingual OCR engines — the swappable last stage

- **Tesseract** (Smith, ICDAR 2007; 100+ language packs, trainable on new scripts).
- **PaddleOCR**: PP-OCRv3 (2206.03001), PaddleOCR 3.0 technical report (2507.05595) — multi-script recognition, separate detection (DB) and recognition models.
- **Surya**, **docTR**, **EasyOCR**: open-source; the Sinhala/Tamil zero-shot study (RANLP 2025, 2507.18264) found Surya best on Sinhala (WER 2.6%) and commercial Document AI best on Tamil, with EasyOCR lacking Sinhala entirely.

**Takeaway:** text *detection* is effectively script-agnostic; *recognition* is the script-specific module, and it is cheap to swap or fine-tune compared to a VLM. Running recognition only on value crops, after structure extraction, minimises the exposure to recognition quality.

### 2.10 Word spotting / query-by-example — the "searchable image" lineage

- Giotis, Sfikas, Gatos & Nikou, *A survey of document image word spotting techniques*, Pattern Recognition 2017 (PDF downloaded from the authors' site).
- **PHOCNet** (1604.00187), **R-PHOC** segmentation-free spotting (1707.01294), Rusakov et al. re-evaluation (1806.10866).

**Relevance:** query-by-example spotting lets a user point at a prompt in one form ("this label") and find the same prompt in other forms *without recognising it*, which is a language-free way to implement "query the form". Combined with a KV linker, it yields "value of this prompt" with no OCR of the key at all. PHOC-style embeddings are also a reasonable way to give the linker a weak, script-agnostic notion of "same label as elsewhere".

### 2.11 Datasets and benchmarks

| Dataset | Size | Languages / scripts | KV links? | Handwriting | Template split | Notes |
|---|---|---|---|---|---|---|
| FUNSD (1905.13538) | 199 forms | EN | yes | some | no | 50-doc test set; noisy labels |
| RFUND (in 2401.03472) | 199 + XFUND | EN + 7 | yes (cleaned) | some | no | use this instead of raw FUNSD |
| XFUND (2104.08836) | 1,393 | ZH JA ES FR IT DE PT | yes | little | no | synthetic-ish fills |
| NAF (1909.02576) | historical forms | EN (no transcripts) | yes | heavy | no | only text-free benchmark |
| SROIE (2103.10213) / CORD | receipts | EN / ID | no (fields) | no | no | KIE, not forms |
| EPHOIE (2102.06732) | 1,494 | ZH | no (fields) | yes | no | |
| Kleister (2105.05796) | long docs | EN | no (fields) | no | no | |
| DocILE (2302.05658) | 6,680 + 100k synth + 932k unlab. | EN | fields + line items | no | **yes** (zero/few/many-shot layouts) | industrial invoices |
| VRDU (2211.15421) | 1,915 + 641 | EN | fields, repeated groups | no | **yes** (STL/MTL/UTL) | unseen-template drop 13–27 F1 |
| KVP10k (2405.00505) | 10,707 pages | mostly EN | yes, incl. unkeyed values & unvalued keys | some | no | fine-tuned Mistral-7B baseline only 0.61–0.66 F1 |
| SIMARA (2304.13606) | archival pages | FR | yes | **full pages** | no | handwritten KV |
| CLEX (2307.10551) | 5,860 | ZH/EN | fields (1,162 types) | no | zero/few-shot | |
| SIBR (in 2304.10759) | 1,000 | ZH/EN | yes | no | no | |
| DKIE (2109.13967) | ~2,500 | ZH | one-shot support/query | no | per-style | mobile captures |
| CommonForms (2509.16506) | 480k pages | 36% non-EN | no (slots only) | no (blank) | by document | blank forms, ideal for synthetic filling |
| AutoFormBench (2603.29832) | 407 | — | no (elements) | no | — | |
| BaFCo (2607.05614) | 200 forms / 1,926 KV | **Bangla** | yes | some | no | only low-resource-script form KIE benchmark found |
| DocVQA (2007.00398) | 12k pages | EN | QA | some | no | query-style evaluation |

**Gap:** there is no multi-script, unseen-template, filled-form KV benchmark. CommonForms (blank, multilingual) + synthetic filling (as in 2608.20868) is the obvious way to build one.

---

## 3. Synthesis — what the evidence establishes

1. **Linking is geometric.** LAGNN (geometry only) ≥ LayoutLMv3 on English linking and ≈2× LiLT zero-shot on JA/ZH; GeoLayoutLM's +9 F1 came from geometric pre-training; FUDGE reaches 57–63 F1 with no text vs 67 for a 138M-param LM. Text semantics resolve a minority of ambiguous links.
2. **Role classification is the language-sensitive step.** Visual-only SER/entity detection trails LM-based by 10–20 points (FUDGE 66.5 vs 78–84; GeoContrastNet 0.65 vs 0.82). Visual appearance (print vs. fill, underline, box) measurably helps (Revising-FUNSD mIoU +0.14 with image), and no paper has yet combined slot/widget detection (CommonForms) with role classification.
3. **Unseen templates, not unseen languages, are the harder generalisation axis** for current models: VRDU shows 13–27 F1 drops from mixed to unseen templates even for LayoutLM/FormNet; KVP10k's cross-template baseline is 0.61–0.66 F1 with a 7B LLM. A geometry-first approach is at least not disadvantaged here, and may be advantaged because it cannot memorise keyword strings (VRDU's main failure mode).
4. **Multilingual evidence is thin.** XFUND = 7 high-resource languages; BaFCo (2026) is the first form-KIE benchmark in a low-resource script; VLM OCR collapses outside ~30 scripts. The proposal addresses a real, documented gap.
5. **Decoder design matters as much as the backbone.** PEneo's joint line-grouping + linking decoding adds ~20 F1 on RFUND for any backbone; LAGNN's rules + ILP add ~1 F1 and enforce well-formed output. Both are backbone-agnostic and should be adopted.
6. **Evaluation hygiene.** FUNSD's test set is 50 documents with inconsistent granularity; LAGNN, FUDGE, GeoLayoutLM and PEneo report *different* RE metrics (word-pair vs entity-level vs pair-extraction). Any comparison must re-run baselines under one metric on RFUND.

---

## 4. Feasibility assessment and recommended direction

### 4.1 Is it doable?

**Yes.** Every component has a published, open-source precedent, and the central hypothesis (structure without language) is already supported by three independent results. The remaining uncertainty is quantitative, not existential: how much role-classification accuracy is lost without text, and whether visual cues + slot detection recover it.

### 4.2 Has anyone done it?

**Partially.** Ranked by closeness:

1. **LAGNN** (Monash, 2023) — language-independent, geometry-only, strong zero-shot cross-lingual. Missing: visual features, blank slots, low-resource scripts, query interface, standard-metric comparison. Code: github.com/Bhanu068/LAGNN.
2. **FUDGE / Davis et al.** (BYU–Adobe, 2019–21) — purely visual, handles handwriting and historical forms, template-free. Missing: multilingual evaluation, weak entity detection. Code: github.com/herobd/FUDGE.
3. **GeoContrastNet** (CVC Barcelona, 2024) — geometry + visual contrastive edges. Weak numbers, multilingual untested.
4. **CommonForms / FFDNet** (2025) — language-agnostic slot detection at scale. No linking.
5. **Revising FUNSD** (2020) — visual key/value segmentation.
6. **LiLT** (2022) — text-optional architecture, but still needs a text encoder.
7. Classic form dropout (1990s) — the pre-printed vs. filled cue.

No published system combines 1 + 4 + a query layer, and none is evaluated on scripts outside XFUND's seven.

### 4.3 What would make it a contribution

- **Visual + geometric role classification** that closes the 10–20 point gap: fuse LAGNN-style edge geometry with (a) FFDNet slot/widget detections, (b) a print-vs-handwriting / filled-vs-blank classifier per box (Davis 2019 already detects this), (c) crop appearance features (FUDGE/GeoContrastNet).
- **Joint grouping + linking decoding** (PEneo) on a text-free backbone, with LAGNN-style ILP constraints for well-formed key→value graphs.
- **Script-agnostic benchmark**: fill CommonForms pages synthetically in multiple scripts (incl. at least one low-resource script, plus handwriting fonts), hold out templates (VRDU-style UTL), and evaluate alongside RFUND, XFUND zero-shot, NAF (no transcripts) and BaFCo.
- **Query layer**: by example (PHOC / crop embedding → find the same prompt → return linked value), by string (OCR keys with a script-specific recogniser), optionally by schema with a small multilingual sentence embedder as an *add-on*, not a dependency.
- **OCR-last design**: recognise only value crops, so recognition quality affects transcription but not structure; report structure metrics independent of OCR (as NAF does).

### 4.4 Risks and how the literature says to mitigate them

| Risk | Evidence | Mitigation |
|---|---|---|
| Role classification ceiling without text (e.g. a value that looks like a key) | FUDGE / GeoContrastNet gaps | Slot detection + print/fill cues; optional tiny text-feature side channel (LiLT-style) that can be dropped for unsupported scripts |
| Tables / repeated groups | DocILE LIR, VRDU line_item ≈ 43–46 F1 for all models | Scope v1 to non-tabular fields; add Form2Seq-style grouping later |
| Small labelled data (FUNSD 149 train docs) | all papers | Synthetic filling of CommonForms; LAGNN shows 139 docs suffice for geometry |
| Text-detector quality on low-resource scripts | 2507.18264, GlotOCR | Use script-agnostic detectors (DB/CRAFT); measure structure metrics with GT boxes and with detected boxes separately (LAGNN/GeoContrastNet both do this) |
| Metric confusion when comparing to prior work | §3.6 | Re-run LAGNN, FUDGE, LiLT, GeoLayoutLM on RFUND under PEneo's pair-extraction F1 |

### 4.5 A minimal first experiment (2–3 weeks)

1. Reproduce LAGNN on RFUND-EN and XFUND zero-shot (code is public), re-scored with PEneo's pair-extraction metric.
2. Add visual node features (ROI crops through a small CNN) and FFDNet slot detections as extra nodes; measure linking and role F1.
3. Run zero-shot on BaFCo (Bangla) and on a synthetic low-resource-script fill of CommonForms; compare to LiLT[InfoXLM] and to one VLM baseline to quantify the gap the proposal claims to close.

If step 2 recovers even half of the role-classification gap while keeping LAGNN's cross-lingual behaviour, the core thesis is established and the rest is system-building and benchmark construction.

---

## 5. Bibliography by folder

See [papers/INDEX.md](papers/INDEX.md) for the auto-generated list with titles, authors, dates, abstracts and file paths. Folder map:

- `01_surveys/` — Ding et al. 2408.01287; Cui et al. 2111.08609; MLLM survey 2507.09861
- `02_datasets_benchmarks/` — FUNSD, Revising-FUNSD, LayoutXLM/XFUND, SROIE, Kleister, DocILE, VRDU, KVP10k, SIMARA, CommonForms, AutoFormBench, BaFCo, PPN/CLEX, DocLayNet, DocVQA
- `03_visual_language_agnostic/` — Davis 2019, FUDGE, **LAGNN**, GeoContrastNet, Doc2Graph, LiLT, VisualWordGrid, Chargrid, Chargrid-OCR, BERTgrid, ViBERTgrid, CUTIE, StrucTexTv2, DiT, Adobe grouping, Form2Seq
- `04_graph_spatial_kv_linking/` — SPADE, Liu-GCN, PICK, DocStruct, GeoLayoutLM, GOSE, KVPFormer, UniVIE, PEneo/RFUND, XFormParser, FormNet, FormNetV2, DocTr, GraphDoc, See-then-Tell, GraphRevisedIE, Identify-Locate-Link, P²Net
- `05_zero_few_shot_template/` — QueryForm, Field-extraction-unlabeled, one-shot graph matching, LF-BP, Attend-Copy-Parse
- `06_layout_pretrained_lms/` — LayoutLM v1–v3, BROS, DocFormer v1–v2, LAMBERT, StrucTexT, UDOP, ERNIE-Layout, TILT, StructuralLM, XYLayoutLM, LayoutReader, VIES/EPHOIE, TRIE
- `07_ocr_free_end_to_end/` — Donut, Pix2Struct, Dessurt
- `08_ocr_engines_multilingual/` — PP-OCRv3, PaddleOCR 3.0, Sinhala/Tamil OCR study, GlotOCR Bench, ThaiOCRBench, Tesseract overview
- `09_word_spotting_query_by_example/` — PHOCNet, R-PHOC, Rusakov re-evaluation, Giotis survey, KV from handwritten pages (2304.13530)

**Not downloaded (paywalled, no preprint found):** Casey et al. 1992; Taylor et al. 1992; Yu & Jain 1996; Cesarini et al. 1998; Lam et al. 1993; Esser et al. 2014 (SciTePress, open on scitepress.org); CORD (Park et al., NeurIPS-W 2019, openreview). All are reachable through CMU library access.
