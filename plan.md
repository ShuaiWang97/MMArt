# PolyArt — ACM MM 2026 Dataset Track

**Working directory:** `/gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/`
**Target venue:** ACM MM 2026 Dataset Track

---

## Project Overview

PolyArt is a multi-perspective art description dataset covering ~75,000 WikiArt paintings.
Each painting is annotated with four independently generated perspectives:

| Perspective | Field | Model | Records | Coverage |
|---|---|---|---|---|
| Narrative | `e_narrative` | Qwen2.5-VL-72B | 75,214 | 99.1% |
| Formal | `e_formal` | GalleryGPT (LLaVA-7B + LoRA) | 75,336 | 99.2% |
| Emotional | `e_emotional` | Qwen3-VL-8B + ArtEmis-v2 | 45,837 | 60.4%† |
| Historical | `e_historical` | Qwen2.5-VL-72B + LightRAG | 75,801 | 99.8% |
| **Unified** | `e_unified` | **Qwen3-8B (vLLM)** | **TODO** | — |

†Coverage limited to paintings with ArtEmis human utterances.

Plus two benchmark experiments validating the dataset:
1. **Perspective Complementarity** — image regeneration across 9 perspective conditions
2. **Reasoning Mode Alignment** — agentic art interpretation across 24 mode–perspective conditions

---

## Current Status

### ✅ Done

| Task | Output | Notes |
|---|---|---|
| Dataset construction | `output/polyart_dataset.json` (195 MB) | 75,336 paintings, 4 perspectives merged |
| Dataset cleaning | `output/polyart_dataset_clean_full4.json` | 74,234 experiment-ready records (all 4 perspectives present and clean) |
| Phase 1: Synthesis prompts | `output/synthesis_prompts/` | 9 conditions × 1,350 paintings stratified sample (50/style × 27 styles) |
| Phase 2: Image regeneration | `output/regenerated_images/` | FLUX.2-Klein-4B + Qwen-Image-2512 × 9 conditions × 1,350 paintings |
| Phase 3: Fidelity evaluation | `output/phase3_results/` | CLIP sim, DINOv2 sim, emotion agreement; results in `results_table.csv` |
| Figures (3) | `output/figures/` + `PolyArt_latex/figures/` | fig1 bar chart, fig2 heatmap, fig3 LOO delta |
| §4 Experiment section | `PolyArt_latex/parts/04-experiment.tex` | §4.1–4.4.1 written; §4.4.2 stub present |
| README | `README.md` | Full project documentation |

### 🔄 In Progress / Next

| Priority | Task | Script | Status |
|---|---|---|---|
| 1 | **e_unified generation** | `scripts/phase4_unified.py` | Script written, not yet run |
| 2 | **Quality evaluation** (BERTScore vs SemArt/ExpArt) | `scripts/quality_eval.py` | Not started — numbers in §4.3 are placeholder |
| 3 | **Task 2: Reasoning Mode Alignment** | TBD | Not started |
| 4 | **Paper writing** | `PolyArt_latex/` | §4 done; §1 intro, §2 related work, §3 method need writing |
| 5 | **HuggingFace upload** | — | Blocked on e_unified |

---

## Phase Details

### Phase 1 — Synthesis Prompts ✅

**Script:** `scripts/phase1_synthesize.py`
**SLURM:** `slurm/run_synthesize.sh`

Generates text prompts for 9 perspective conditions on the 1,350-painting stratified sample:
- **Singles (4):** N, F, E, H — copied directly from the clean dataset
- **Leave-one-out (4):** NFE, NFH, NEH, FEH — synthesized with Qwen3-8B (vLLM, ~80 words)
- **Full (1):** NFEH — synthesized with Qwen3-8B (vLLM, ~80 words)

Output schema per condition: `output/synthesis_prompts/{CONDITION}/synthesis_{CONDITION}.json`

---

### Phase 2 — Image Regeneration ✅

**Script:** `scripts/phase2_image_regen.py`
**SLURM:** `slurm/run_regen.sh`

Two T2I models used to confirm findings are model-agnostic:

| Model | Pipeline | Steps | Resolution |
|---|---|---|---|
| FLUX.2-Klein-4B | `Flux2KleinPipeline` | 4 steps, guidance_scale=1.0 | 1024 → 512 |
| Qwen-Image-2512 | `DiffusionPipeline` | 25 steps, true_cfg_scale=4.0 | 1024 → 512 |

Output: `output/regenerated_images/{model}/{condition}/{style}__{filename}.jpg`

---

### Phase 3 — Fidelity Evaluation ✅

**Script:** `scripts/phase3_eval.py`
**SLURM:** `slurm/run_eval.sh` (`-p gpu_h100`)

Three metrics per (painting, condition, model) triple:

| Metric | Model | Captures |
|---|---|---|
| CLIP cosine similarity | CLIP ViT-L/14 | Style / semantic fidelity |
| DINOv2 cosine similarity | DINOv2-Large CLS token | Compositional / structural fidelity |
| Emotion agreement | CLIP zero-shot vs. 9 ARTEMIS labels | Affective fidelity (top-1 match) |

**Key results** (averaged across FLUX.2-Klein and Qwen-Image):

| Condition | CLIP ↑ | DINOv2 ↑ | Emotion ↑ |
|---|---|---|---|
| N (Narrative) | 0.578 | 0.344 | 0.224 |
| F (Formal) | **0.641** | 0.401 | 0.343 |
| E (Emotional) | 0.612 | **0.416** | 0.257 |
| H (Historical) | 0.603 | 0.378 | **0.674** |
| NFE | 0.647 | 0.456 | 0.320 |
| NFH | 0.691 | 0.505 | 0.457 |
| NEH | 0.682 | 0.498 | 0.439 |
| FEH | 0.692 | 0.479 | 0.491 |
| NFEH (full) | 0.689 | 0.503 | 0.431 |

**Findings:**
- F → CLIP (style); E → DINOv2 (composition); H → Emotion (affective tone) — clear diagonal alignment
- NFEH > best single on CLIP and DINOv2
- H alone (0.674) > NFEH (0.431) on emotion — synthesis dilutes concentrated affective signal

Figures: `output/figures/fig1_conditions_bar.pdf`, `fig2_alignment_heatmap.pdf`, `fig3_loo_delta.pdf`

---

### Phase 4 — e_unified Generation 🔄

**Script:** `scripts/phase4_unified.py`
**SLURM:** `slurm/run_unified.sh` (16h, gpu_h100, 60G)

Generates ~150-word harmonized unified captions for all 74,234 paintings using Qwen3-8B (vLLM).

- **~150 words** (not 80w like Phase 1 prompts) — designed for retrieval/LLM tasks, not T2I
- Both FLUX.2-Klein and Qwen-Image use T5/custom text encoders (not CLIP) — no 77-token constraint
- System prompt: integrate all 4 perspectives into coherent prose, present tense, no headers

Two outputs:
1. `output/polyart_dataset_unified.json` — full dataset + `e_unified` field (HuggingFace deliverable)
2. `output/synthesis_prompts/U/synthesis_U.json` — condition U for Phase 3 as 10th condition

**To run:**
```bash
# Quick test on sample only
python scripts/phase4_unified.py --mode sample --batch_size 32

# Full job
sbatch slurm/run_unified.sh
```

After phase4 completes: re-run Phase 2 + Phase 3 for condition U, add U row to results table and paper.

---

### Quality Evaluation — Gold Subset 📋

**Script:** `scripts/quality_eval.py` (not yet written)

BERTScore F1 between generated perspectives and human-authored references on ~500 paintings
overlapping with SemArt and ExpArt:

| Perspective | Reference | Target BERTScore |
|---|---|---|
| `e_narrative` | SemArt commentaries | — |
| `e_formal` | ExpArt explanations | — |
| `e_emotional` | ArtEmis utterances (avg 5.7/painting) | — |
| `e_historical` | SemArt commentaries | — |
| `e_unified` | SemArt commentaries (holistic) | — |

Also: hallucination rate for `e_historical` (artist name, movement, period vs. WikiArt/Wikidata metadata).

**Note:** Numbers currently in §4.3 are placeholders — need to run this before submission.

---

### Task 2 — Reasoning Mode Alignment 📋

**Script:** TBD
**Described in:** `PolyArt_latex/parts/04-experiment.tex` §4.4.2 (stub)

24 conditions: 6 reasoning modes × 4 perspectives, on 300-painting gold subset.

| Reasoning Mode | Natural Perspective Alignment |
|---|---|
| Formal | `e_formal` |
| Causal | `e_historical` |
| Semiotic | `e_historical`, `e_narrative` |
| Phenomenological | `e_emotional` |
| Analogical | cross-perspective |
| Counterfactual | cross-perspective |

Agentic system: Claude Sonnet conditioned on image + metadata + Wikidata context.
Evaluation: BERTScore vs. SemArt/ExpArt, LLM-as-judge (perspective fidelity, coherence), specificity.

**Dependency:** Needs quality evaluation (gold subset) to be done first for reference texts.

---

## Dataset Outputs

| File | Contents | Status |
|---|---|---|
| `output/polyart_dataset.json` | Raw merged dataset, 75,336 paintings | ✅ |
| `output/polyart_dataset_clean.json` | Cleaned, nulls preserved, 75,336 paintings | ✅ |
| `output/polyart_dataset_clean_full4.json` | Experiment-ready, all 4 perspectives present, 74,234 paintings | ✅ |
| `output/polyart_dataset_unified.json` | Full dataset + `e_unified` field | 🔄 Phase 4 |

**HuggingFace upload:** blocked on `polyart_dataset_unified.json`. Two-tier design:
- Full split (75,336): includes nulled perspectives, `e_unified` where available
- Experiment split (74,234): all 4 perspectives clean, `e_unified` present

---

## LaTeX Paper Status

| Section | Status | Notes |
|---|---|---|
| §1 Introduction | ❌ Not written | |
| §2 Related Work | ❌ Not written | |
| §3 Dataset Construction | ❌ Not written | Pipeline well-documented, just needs writing |
| §4.1 Dataset Statistics | ✅ Written | Table: generation coverage; top-10 styles |
| §4.2 Perspective Diversity | ✅ Written | Lexical diversity; pairwise CLIP similarity |
| §4.3 Gold Subset Quality | ⚠️ Written with placeholder numbers | Needs quality_eval.py to run |
| §4.4.1 Task 1: Complementarity | ✅ Written | Tables + 3 figures |
| §4.4.2 Task 2: Reasoning Alignment | ⚠️ Stub only | Not implemented yet |
| §5 Use Cases / Applications | ❌ Not written | |

---

## Infrastructure Notes

- **Cluster partitions:** `gpu_h100`, `gpu_a100` (not `gpu`)
- **SLURM GPU flag:** `--gres=gpu:1` (not `--gpus=1`)
- **Conda env:** `gallery_gpt` — always `source activate gallery_gpt`
- **Module loads:** `module purge && module load 2023 && module load Anaconda3/2023.07-2`
- **Python for vLLM:** `/home/wangsh/.conda/envs/gallery_gpt/bin/python`
- **WikiArt images:** `/gpfs/work5/0/prjs0996/data/wikiart/Images/`
- **sklearn fix if needed:** `pip install -U scikit-learn` (numpy binary incompatibility)
