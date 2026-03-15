# PolyArt: A Multi-Perspective Dataset for Art Understanding

**PolyArt** is a large-scale dataset of ~75,000 WikiArt paintings, each annotated with four independent expert-style perspectives: narrative scene description, formal visual analysis, emotional response, and historical context. The dataset is designed to study whether different interpretive lenses encode genuinely distinct visual information.

> Target venue: **ACM MM 2026 Dataset Track**

---

## The Four Perspectives

| Key | Perspective | What it captures | Model |
|---|---|---|---|
| `e_narrative` | Narrative & Scene | Depicted entities, figures, scene, story | Qwen2.5-VL-72B-Instruct |
| `e_formal` | Formal Analysis | Composition, brushwork, palette, light/shadow | GalleryGPT (LLaVA-7B + LoRA) |
| `e_emotional` | Emotional Response | Mood, atmosphere, psychological tone | Qwen3-VL-8B + ARTEMIS-v2 |
| `e_historical` | Historical Context | Art-historical meaning, symbolism, cultural codes | Qwen2.5-VL-72B + LightRAG KG |

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total paintings | 75,336 |
| Art styles | 27 (WikiArt taxonomy) |
| Artists | ~1,111 |
| Full coverage (all 4 perspectives) | 74,234 (98.5%) |
| Average caption length | ~70–80 words per perspective |

---

## Repository Structure

```
PolyArt/
├── scripts/
│   ├── generate_perspectives.py   # Main VLM generation pipeline (all 4 perspectives)
│   ├── consolidate.py             # Merge 32 job files → polyart_dataset.json
│   ├── clean_dataset.py           # Quality cleaning → polyart_dataset_clean.json
│   ├── phase1_synthesize.py       # Pre-generate prompts for image regeneration experiment
│   ├── sample_data.py             # Stratified sampling utilities
│   ├── MLLM_inference.py          # Low-level VLM inference utilities
│   ├── substract_artremis2.py     # ARTEMIS-v2 utterance processing
│   └── save_image.py              # Image handling utilities
├── slurm/
│   ├── run_perspectives.sh        # Submit 8-job parallel SLURM array per perspective
│   └── run_synthesize.sh          # Submit synthesis prompt generation job
├── GalleryGPT/                    # LLaVA-7B + LoRA for formal analysis (ACM MM 2024)
├── ArtRAG/
│   └── art_context/               # Pre-built LightRAG knowledge graph (61 MB)
│       ├── kv_store_full_docs.json
│       ├── kv_store_text_chunks.json
│       ├── vdb_chunks.json
│       └── graph_chunk_entity_relation.graphml
├── output/
│   ├── e_narrative/               # narrative_job_{0-7}.json       (raw generation)
│   ├── formal_analysis/           # llava_captions_job_{0-7}.json  (raw generation)
│   ├── e_emotional/               # emotional_job_{0-7}.json       (raw generation)
│   ├── e_historical/              # historical_job_{0-7}.json      (raw generation)
│   ├── artemis-v2/                # ARTEMIS-v2 emotional utterances (292 MB)
│   ├── polyart_dataset.json                # Consolidated (pre-cleaning)
│   ├── polyart_dataset_clean.json          # Tier-1: full 75,336 records
│   ├── polyart_dataset_clean_full4.json    # Tier-2: 74,234 complete records
│   ├── polyart_dataset_clean.parquet       # HuggingFace-ready (72 MB)
│   ├── cleaning_report.json                # Per-step quality audit
│   └── synthesis_prompts/         # Pre-generated prompts for experiments
├── PolyArt_latex/                 # ACM MM paper (LaTeX)
└── plan.md                        # Detailed project plan and research questions
```

---

## Data Assets (external)

| Asset | Path | Size |
|---|---|---|
| WikiArt images | `/gpfs/work5/0/prjs0996/data/wikiart/Images/` | 26 GB |
| WikiArt metadata | `/gpfs/work5/0/prjs0996/data/wikiart/wikiart_full.csv` | 18 MB |
| CLIP embeddings (ViT-L/14) | `/gpfs/work5/0/prjs0996/data/wikiart/simp_wikiart_Vit-L.npz` | 159 MB |

---

## Dataset Schema

Each record in `polyart_dataset_clean.json`:

```json
{
  "image_id":         "Romanticism/delacroix_liberty-leading-the-people.jpg",
  "title":            "liberty-leading-the-people.jpg",
  "artist":           "delacroix",
  "style":            "romanticism",
  "date":             "1830",
  "e_narrative":      "A triumphant woman strides forward...",
  "e_formal":         "The diagonal composition surges from lower left...",
  "e_emotional":      "The painting radiates fierce exhilaration...",
  "e_historical":     "Painted in the wake of the July Revolution...",
  "dominant_emotion": "awe",
  "artemis_coverage": true,
  "rag_sim":          0.74,
  "n_perspectives":   4
}
```

| Field | Type | Description |
|---|---|---|
| `image_id` | str | WikiArt relative path (unique key) |
| `e_narrative` | str \| null | Narrative scene description |
| `e_formal` | str \| null | Formal visual analysis |
| `e_emotional` | str \| null | Emotional response |
| `e_historical` | str \| null | Historical context |
| `dominant_emotion` | str \| null | Majority-vote emotion from ARTEMIS-v2 |
| `artemis_coverage` | bool \| null | True if ARTEMIS utterances were available |
| `rag_sim` | float \| null | Cosine similarity of best LightRAG retrieval |
| `n_perspectives` | int | Count of non-null perspectives (0–4) |

---

## Reproduction

### Step 1 — Generate perspectives (SLURM)

```bash
# Submit 8 parallel jobs per perspective (~18h wall clock on H100s)
bash slurm/run_perspectives.sh narrative
bash slurm/run_perspectives.sh formal
bash slurm/run_perspectives.sh emotional
bash slurm/run_perspectives.sh historical
```

Or run a single job manually:

```bash
conda activate gallery_gpt
python scripts/generate_perspectives.py --perspective narrative --job_id 0
```

### Step 2 — Consolidate 32 job files into one dataset

```bash
python scripts/consolidate.py
# Output: output/polyart_dataset.json       (~195 MB)
#         output/polyart_dataset.jsonl
#         output/polyart_dataset.parquet
```

### Step 3 — Clean and quality-filter

```bash
python scripts/clean_dataset.py
# Output: output/polyart_dataset_clean.json         (full 75,336 records)
#         output/polyart_dataset_clean_full4.json    (74,234 complete records)
#         output/polyart_dataset_clean.parquet       (HuggingFace-ready, 72 MB)
#         output/cleaning_report.json
```

Four cleaning steps are applied automatically:

| Step | Action | Affected |
|---|---|---|
| 1. CJK contamination | Null captions containing Chinese/Japanese/Korean characters | 1,003 |
| 2. Short captions | Null captions under 20 words (likely model errors) | 7 |
| 3. Overlong `e_formal` | Truncate at sentence boundary ≤ 150 words | 20,267 |
| 4. Duplicate `e_formal` | Null exact duplicates (generic fallback responses) | 93 |

**Two-tier output design:**
- `polyart_dataset_clean.json` — all 75,336 paintings with `null` for any removed perspective. Honest and complete for dataset users.
- `polyart_dataset_clean_full4.json` — only the 74,234 paintings where all four perspectives are non-null. Used as the experiment pool.

### Step 4 — Pre-generate synthesis prompts (experiment)

```bash
bash slurm/run_synthesize.sh
# Or: python scripts/phase1_synthesize.py
# Output: output/synthesis_prompts/{N,F,E,H,NFE,NFH,NEH,FEH,NFEH}/
```

Generates text prompts for the perspective complementarity experiment across 9 conditions on a stratified 1,000-painting sample. Singles reuse existing captions directly; multi-perspective conditions are synthesized with Qwen3-8B via vLLM.

---

## Experiment: Perspective Complementarity

The core experiment tests whether each perspective encodes distinct visual information by regenerating images from text prompts and measuring reconstruction fidelity against the original painting.

**9 conditions** (reduced from the 15 full power-set subsets):

| Type | Conditions | Purpose |
|---|---|---|
| Singles (4) | N, F, E, H | Baseline: what does each perspective alone recover? |
| Leave-one-out (4) | NFE, NFH, NEH, FEH | Marginal contribution: what is lost when one perspective is absent? |
| Full (1) | NFEH | Upper bound: all perspectives combined |

**How multi-perspective conditions are combined:** Each subset of perspectives is synthesized into a single coherent ~80-word prompt using Qwen3-8B, so all 9 conditions enter the image generator in the same text format.

**Evaluation dimensions** (regenerated image vs. original):

| Metric | Measures |
|---|---|
| CLIP style similarity | Style and period fidelity |
| DINO composition score | Spatial and compositional fidelity |
| VQA content accuracy | Iconographic content fidelity |
| Emotion classifier agreement | Affective tone fidelity |

**Hypothesis:** Each perspective recovers different dimensions; removing any one degrades at least one metric; NFEH achieves the best overall reconstruction.

---

## Quality Validation

Three levels of validation are planned:

1. **BERTScore vs. SemArt / ExpArt** — ~500 paintings overlapping with human-written expert commentary. Primary quantitative result in the paper.
2. **LLM-as-judge** — Perspective fidelity scoring (1–5) across all 75k paintings. Supplementary quality scan.
3. **Human expert study** — 3 annotators × 100 paintings × 4 perspectives. Gold standard for reviewer credibility.

---

## Key Quality Notes

- **`e_formal` CJK contamination (1.1%):** GalleryGPT (LLaVA-7B) defaults to Chinese for visually ambiguous or abstract paintings. Detected via Unicode range `\u4e00–\u9fff` and nulled at cleaning step 1.
- **`e_formal` caption length:** GalleryGPT has no length constraint; average is 135 words vs. ~70 words for other perspectives. Captions over 150 words are truncated at the nearest sentence boundary.
- **Perspective vocabulary overlap:** 42% of `e_formal` captions contain emotion-related words and 37% of `e_emotional` captions contain technique-related words. This reflects natural art-writing vocabulary, not model confusion.
