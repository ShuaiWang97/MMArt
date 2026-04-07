# MMArt: A Multi-Perspective Multimodal Dataset for Visual Art Understanding

**MMArt** is a large-scale dataset of 74,234 WikiArt paintings, each annotated with four independent expert-style perspectives: narrative scene description, formal visual analysis, emotional response, and historical context. The dataset is designed to study whether different interpretive lenses encode genuinely distinct visual information.

---

## The Four Perspectives

| Key | Perspective | What it captures | Model |
|---|---|---|---|
| `e_narrative` | Narrative & Scene | Depicted entities, figures, scene, story | Qwen3-VL-8B-Instruct |
| `e_formal` | Formal Analysis | Composition, brushwork, palette, light/shadow | GalleryGPT (LLaVA-7B + LoRA) |
| `e_emotional` | Emotional Response | Mood, atmosphere, psychological tone | Qwen3-VL-8B-Instruct + ARTEMIS-v2 |
| `e_historical` | Historical Context | Art-historical meaning, symbolism, cultural codes | RAG with Art context knowledge |

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total paintings | 74,234 |
| Art styles | 20 (WikiArt taxonomy) |
| Artists | 743 |
| Text fields per painting | 5 (N, F, E, H + Unified) |
| Average caption length | ~70–80 words per perspective |

---

## Output Dataset Files

| File | Records | Description |
|---|---|---|
| `MMArt_dataset.json` | 75,336 | Raw consolidated dataset — all paintings with 4 perspectives (N, F, E, H). Starting point before any cleaning. |
| `MMArt_dataset_clean.json` | 75,336 | After cleaning — CJK contamination nulled, short/truncated descriptions fixed. Record count unchanged (entries repaired, not dropped). |
| `MMArt_dataset_clean_full4.json` | 74,234 | **Experiment pool** — 1,102 paintings dropped where any perspective remained null after cleaning. Used as the retrieval gallery. |
| `MMArt_dataset_unified.json` | 74,234 | **Most complete version** — same as `clean_full4` with one extra field `e_unified` (Phase 4 LLM-synthesized unified caption). |
| `MMArt_dataset_unified.parquet` | 74,234 | HuggingFace-ready Parquet version of the unified dataset (108 MB). |
| `cleaning_report.json` | — | Audit log from the cleaning pipeline: per-step counts of removed/repaired records with affected `image_id` lists. |

---

## Repository Structure

```
MMArt/
├── scripts/
│   ├── generate_perspectives.py   # Main VLM generation pipeline (all 4 perspectives)
│   ├── consolidate.py             # Merge job files → MMArt_dataset.json
│   ├── clean_dataset.py           # Quality cleaning → MMArt_dataset_clean.json
│   ├── phase1_synthesize.py       # Pre-generate prompts for image regeneration experiment
│   ├── phase2_image_regen.py      # Image regeneration (FLUX.2-Klein, Qwen-Image)
│   ├── phase4_unified.py          # Unified caption generation (Qwen3-8B via vLLM)
│   ├── quality_eval.py            # LLM-as-judge quality evaluation
│   ├── phase5_retrieval.py        # Text-to-image retrieval benchmark
│   ├── upload_to_huggingface.py   # Upload dataset to HuggingFace Hub
│   └── make_composites.py         # Generate qualitative composite images
├── slurm/
│   ├── run_perspectives.sh        # Submit 8-job parallel SLURM array per perspective
│   ├── run_synthesize.sh          # Submit synthesis prompt generation job
│   ├── run_image_regen.sh         # Submit image regeneration jobs
│   ├── run_unified.sh             # Submit unified caption generation job
│   └── run_quality_eval.sh        # Submit quality evaluation job
├── dataset/                       # Final dataset files (uploaded to HuggingFace)
├── docs/                          # Supplementary website (ACM MM requirement)
│   └── index.html
└── GalleryGPT/                    # LLaVA-7B + LoRA for formal analysis (ACM MM 2024)
```

---

## Data Assets (external, not in repo)

| Asset | Path | Size |
|---|---|---|
| WikiArt images | `/gpfs/work5/0/prjs0996/data/wikiart/Images/` | 26 GB |
| WikiArt metadata | `/gpfs/work5/0/prjs0996/data/wikiart/wikiart_full.csv` | 18 MB |
| CLIP embeddings (ViT-L/14) | `/gpfs/work5/0/prjs0996/data/wikiart/simp_wikiart_Vit-L.npz` | 159 MB |

---

## Dataset Schema

Each record in `MMArt_dataset_unified.json`:

```json
{
  "image_id":         "Romanticism/delacroix_liberty-leading-the-people.jpg",
  "title":            "liberty-leading-the-people",
  "artist":           "delacroix",
  "style":            "romanticism",
  "date":             "1830",
  "e_narrative":      "A triumphant woman strides forward...",
  "e_formal":         "The diagonal composition surges from lower left...",
  "e_emotional":      "The painting radiates fierce exhilaration...",
  "e_historical":     "Painted in the wake of the July Revolution...",
  "e_unified":        "Liberty Leading the People captures the explosive...",
  "dominant_emotion": "awe",
  "artemis_coverage": true,
  "rag_sim":          0.74,
  "n_perspectives":   4
}
```

| Field | Type | Description |
|---|---|---|
| `image_id` | str | WikiArt relative path (unique key) |
| `e_narrative` | str \| null | Narrative scene description (~80 words) |
| `e_formal` | str \| null | Formal visual analysis (~80 words) |
| `e_emotional` | str \| null | Emotional response (~80 words) |
| `e_historical` | str \| null | Historical context (~80 words) |
| `e_unified` | str \| null | Unified caption integrating all four perspectives (~150 words) |
| `dominant_emotion` | str \| null | Majority-vote emotion from ARTEMIS-v2 |
| `artemis_coverage` | bool \| null | True if ARTEMIS utterances were available |
| `rag_sim` | float \| null | Cosine similarity of best RAG retrieval hit |
| `n_perspectives` | int | Count of non-null perspectives (0–4) |

---

## Reproduction

### Step 1 — Generate perspectives (SLURM)

```bash
# Submit 8 parallel jobs per perspective
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

### Step 2 — Consolidate job files into one dataset

```bash
python scripts/consolidate.py
# Output: output/MMArt_dataset.json
#         output/MMArt_dataset.jsonl
```

### Step 3 — Clean and quality-filter

```bash
python scripts/clean_dataset.py
# Output: output/MMArt_dataset_clean.json         (full 75,336 records)
#         output/MMArt_dataset_clean_full4.json    (74,234 complete records)
#         output/cleaning_report.json
```

Four cleaning steps are applied automatically:

| Step | Action | Affected |
|---|---|---|
| 1. CJK contamination | Null captions containing Chinese/Japanese/Korean characters | 1,003 |
| 2. Short captions | Null captions under 20 words (likely model errors) | 7 |
| 3. Overlong `e_formal` | Truncate at sentence boundary ≤ 150 words | 20,267 |
| 4. Duplicate `e_formal` | Null exact duplicates (generic fallback responses) | 93 |

### Step 4 — Generate unified captions (Phase 4)

```bash
bash slurm/run_unified.sh
# Or: python scripts/phase4_unified.py
# Output: output/MMArt_dataset_unified.json
```

### Step 5 — Pre-generate synthesis prompts (reconstruction experiment)

```bash
bash slurm/run_synthesize.sh
# Or: python scripts/phase1_synthesize.py
# Output: output/synthesis_prompts/{N,F,E,H,NFE,NFH,NEH,FEH,NFEH}/
```

### Step 6 — Image regeneration

```bash
bash slurm/run_image_regen.sh
# Runs FLUX.2-Klein-4B and Qwen-Image-2512 across all 9 conditions
```

---

## Experiment: Perspective Complementarity

The core experiment tests whether each perspective encodes distinct visual information by regenerating images from text prompts and measuring reconstruction fidelity against the original painting.

**9 conditions:**

| Type | Conditions | Purpose |
|---|---|---|
| Singles (4) | N, F, E, H | Baseline: what does each perspective alone recover? |
| Leave-one-out (4) | NFE, NFH, NEH, FEH | Marginal contribution: what is lost when one perspective is absent? |
| Full (1) | NFEH | Upper bound: all perspectives combined |

**Evaluation dimensions** (regenerated image vs. original):

| Metric | Measures |
|---|---|
| CLIP style similarity | Style and period fidelity |
| DINOv3 composition score | Spatial and compositional fidelity |
| Emotion classifier agreement | Affective tone fidelity |

---

## Key Quality Notes

- **`e_formal` CJK contamination (1.1%):** GalleryGPT (LLaVA-7B) defaults to Chinese for visually ambiguous or abstract paintings. Detected via Unicode range `\u4e00–\u9fff` and nulled at cleaning step 1.
- **`e_formal` caption length:** GalleryGPT has no length constraint; average is 135 words vs. ~70 words for other perspectives. Captions over 150 words are truncated at the nearest sentence boundary.
- **ARTEMIS-v2 coverage:** 99.0% of paintings have emotional grounding from ARTEMIS-v2 human utterances. The remaining 1% use a vision-only fallback prompt.
- **RAG coverage:** Historical perspective uses sentence-transformers (`all-MiniLM-L6-v2`) retrieval over Wikipedia art pages. Paintings with max cosine similarity < 0.25 fall back to a no-RAG prompt.
