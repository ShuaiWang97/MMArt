"""
phase4_unified.py

Generates e_unified captions for the PolyArt dataset using Qwen3-8B
via vLLM batched inference.

e_unified is a ~150-word harmonized description that integrates all four
perspectives (narrative, formal, emotional, historical) into coherent prose.
Unlike the ~80-word synthesis prompts used in Phase 1 (optimised for T2I
generation), e_unified is designed for downstream retrieval and LLM tasks
where richer text is beneficial.

Two outputs:
  1. Full dataset  → output/polyart_dataset_unified.json
     polyart_dataset_clean_full4.json with e_unified field added.

  2. Sample condition U → output/synthesis_prompts/U/synthesis_U.json
     e_unified for the Phase 3 sample paintings, formatted identically to
     other synthesis conditions so it slots into the Phase 2/3 pipeline as
     a 10th condition.

Usage:
  python scripts/phase4_unified.py
  python scripts/phase4_unified.py --batch_size 64
  python scripts/phase4_unified.py --mode sample   # Phase 3 sample only (fast test)
  python scripts/phase4_unified.py --mode full     # Full dataset only

SLURM:
  sbatch slurm/run_unified.sh
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).resolve().parent.parent
DATASET_FILE = REPO_ROOT / "output" / "polyart_dataset_clean_full4.json"
PROMPTS_DIR  = REPO_ROOT / "output" / "synthesis_prompts"
OUT_DATASET  = REPO_ROOT / "output" / "polyart_dataset_unified.json"
OUT_SAMPLE   = PROMPTS_DIR / "U"
CHECKPOINT   = REPO_ROOT / "output" / "phase4_checkpoint.json"

MODEL_ID     = "Qwen/Qwen3-8B"
MAX_TOKENS   = 250   # ~150 words + buffer for tokenisation overhead
TARGET_WORDS = 150

PERSPECTIVE_FULL = {
    "N": "e_narrative",
    "F": "e_formal",
    "E": "e_emotional",
    "H": "e_historical",
}

PERSPECTIVE_LABELS = {
    "N": "Narrative — what is depicted, scene and subjects",
    "F": "Formal — composition, palette, technique, brushwork",
    "E": "Emotional — mood, atmosphere, affective tone",
    "H": "Historical — art movement, period, cultural context",
}

SYSTEM_PROMPT = """\
You are an art writer producing unified painting descriptions for an academic dataset.

Given four analytical perspectives on a painting, write a single unified \
description of approximately 150 words that integrates all four perspectives \
into coherent prose.

Rules:
- Do not use section headers or bullet points
- Do not start with "This painting" or "The painting"
- Write in present tense, third person
- Preserve specific details from each perspective: what is depicted, \
visual structure and technique, emotional atmosphere, and art-historical context
- Output only the description, nothing else"""


def build_user_prompt(record: dict) -> str:
    title  = record.get("title", "Untitled")
    artist = record.get("artist", "Unknown")
    lines  = [f'Painting: "{title}" by {artist}\n']
    for key, label in PERSPECTIVE_LABELS.items():
        field = PERSPECTIVE_FULL[key]
        cap   = record.get(field, "")
        if cap:
            lines.append(f"[{label}]\n{cap}")
    lines.append(f"\nWrite a unified ~{TARGET_WORDS}-word description integrating all perspectives above.")
    return "\n\n".join(lines)


def build_prompt_text(record: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(record)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # disable Qwen3 chain-of-thought
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: {len(ckpt):,} records done")
        return ckpt
    return {}


def save_checkpoint(results: dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(results, f)


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def generate_unified(
    records: list[dict],
    llm,
    tokenizer,
    sampling_params,
    batch_size: int,
    existing: dict,
) -> dict:
    """
    Generate e_unified for a list of records.
    Returns {image_id: e_unified_text}.
    """
    results  = dict(existing)
    todo     = [r for r in records if r.get("image_id") not in results]
    print(f"Records to generate: {len(todo):,}  (already done: {len(results):,})")

    for start in tqdm(range(0, len(todo), batch_size), desc="Generating e_unified"):
        batch = todo[start : start + batch_size]

        prompts    = []
        valid_recs = []
        for r in batch:
            # Skip records missing all perspectives
            if not any(r.get(PERSPECTIVE_FULL[k]) for k in "NFEH"):
                continue
            prompts.append(build_prompt_text(r, tokenizer))
            valid_recs.append(r)

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params)

        for rec, out in zip(valid_recs, outputs):
            results[rec["image_id"]] = out.outputs[0].text.strip()

        save_checkpoint(results)
        print(f"  Checkpoint: {len(results):,} total")

    return results


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_full_dataset(dataset: list[dict], results: dict) -> None:
    for rec in dataset:
        rec["e_unified"] = results.get(rec["image_id"])

    with open(OUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    n_filled = sum(1 for r in dataset if r.get("e_unified"))
    print(f"Full dataset saved → {OUT_DATASET}")
    print(f"  e_unified present: {n_filled:,} / {len(dataset):,}")


def write_sample_condition(results: dict, dataset_index: dict, sample_df: pd.DataFrame) -> None:
    """
    Write condition U in the same schema as other synthesis conditions so it
    slots directly into the Phase 2/3 pipeline.
    """
    OUT_SAMPLE.mkdir(parents=True, exist_ok=True)
    out_path = OUT_SAMPLE / "synthesis_U.json"

    records = []
    missing = 0
    for _, row in sample_df.iterrows():
        iid     = row["image_id"]
        unified = results.get(iid)
        if unified is None:
            print(f"  Warning: no e_unified for {iid}")
            missing += 1
            continue
        records.append({
            "image_id":            iid,
            "title":               row.get("title", ""),
            "artist":              row.get("artist", ""),
            "style":               row.get("style", ""),
            "date":                str(row.get("date", "")),
            "condition":           "U",
            "perspectives_used":   list(PERSPECTIVE_FULL.values()),
            "synthesized_caption": unified,
        })

    pd.DataFrame(records).to_json(out_path, orient="records", indent=2)
    print(f"Sample condition U saved → {out_path}")
    print(f"  {len(records)} records  ({missing} missing)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: e_unified generation")
    parser.add_argument(
        "--mode", choices=["all", "full", "sample"], default="all",
        help="'all' (default): full dataset + sample; 'full': dataset only; 'sample': sample only",
    )
    parser.add_argument("--batch_size",            type=int,   default=64)
    parser.add_argument("--model_name",            type=str,   default=MODEL_ID)
    parser.add_argument("--gpu_mem",               type=float, default=0.85)
    parser.add_argument("--n_sample",              type=int,   default=1350,
                        help="Expected sample size; used to find the sample CSV (default: 1350)")
    args = parser.parse_args()

    # --- Load dataset ---
    print(f"Loading dataset: {DATASET_FILE} ...")
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  {len(dataset):,} records loaded")

    # Build image_id → record index for sample lookups
    dataset_index = {r["image_id"]: r for r in dataset}

    # --- Resolve which records need generation ---
    if args.mode == "sample":
        # Only generate for the sample paintings
        sample_csv = _find_sample_csv(args.n_sample)
        sample_df  = pd.read_csv(sample_csv)
        print(f"Sample mode: {len(sample_df)} paintings from {sample_csv}")
        records_to_gen = [dataset_index[iid] for iid in sample_df["image_id"]
                          if iid in dataset_index]
    else:
        # Full dataset
        records_to_gen = dataset
        sample_csv     = _find_sample_csv(args.n_sample)
        sample_df      = pd.read_csv(sample_csv) if sample_csv else None

    # --- Load vLLM ---
    print(f"\nLoading {args.model_name} via vLLM ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        repetition_penalty=1.05,
    )
    print("  Model loaded.\n")

    # --- Generate ---
    existing = load_checkpoint()
    results  = generate_unified(
        records_to_gen, llm, tokenizer, sampling_params, args.batch_size, existing
    )

    # --- Write outputs ---
    if args.mode in ("all", "full"):
        write_full_dataset(dataset, results)

    if args.mode in ("all", "sample") and sample_df is not None:
        write_sample_condition(results, dataset_index, sample_df)
    elif args.mode == "sample":
        print("Warning: no sample CSV found; skipping condition U output.")

    # Clean up checkpoint
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("\nCheckpoint removed.")

    print("\nPhase 4 complete.")


def _find_sample_csv(n_sample: int) -> Path | None:
    """Find sample CSV — tries exact n_sample first, then scans for any sample_*.csv."""
    exact = PROMPTS_DIR / f"sample_{n_sample}.csv"
    if exact.exists():
        return exact
    candidates = sorted(PROMPTS_DIR.glob("sample_*.csv"))
    if candidates:
        found = candidates[-1]
        print(f"  Note: sample_{n_sample}.csv not found; using {found.name}")
        return found
    print(f"  Warning: no sample CSV found in {PROMPTS_DIR}")
    return None


if __name__ == "__main__":
    main()
