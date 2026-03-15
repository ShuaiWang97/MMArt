"""
phase1_synthesize.py

Pre-generates synthesized text prompts for the perspective complementarity experiment.
Covers 9 conditions:

  Singles (4):       N, F, E, H         — copied directly from existing outputs (no LLM)
  Leave-one-out (4): NFE, NFH, NEH, FEH — synthesized with Qwen3-8B via vLLM
  Full (1):          NFEH               — synthesized with Qwen3-8B via vLLM

Usage:
  python scripts/phase1_synthesize.py                        # all conditions, 1000 paintings
  python scripts/phase1_synthesize.py --condition NFE        # single condition
  python scripts/phase1_synthesize.py --n_sample 200        # quick test

Outputs (one file per condition):
  output/synthesis_prompts/<CONDITION>/synthesis_<CONDITION>.json

Output schema per record:
  {
    "image_id":            str,
    "title":               str,
    "artist":              str,
    "style":               str,
    "date":                str,
    "condition":           str,   # e.g. "NFE"
    "perspectives_used":   [str], # e.g. ["e_narrative", "e_formal", "e_emotional"]
    "synthesized_caption": str,
  }

The same 1000-painting sample is reused across all conditions (saved to
output/synthesis_prompts/sample_1000.csv on first run).
"""

import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full perspective key names (for output schema)
PERSPECTIVE_FULL = {
    "N": "e_narrative",
    "F": "e_formal",
    "E": "e_emotional",
    "H": "e_historical",
}

# Human-readable labels for the synthesis prompt
PERSPECTIVE_LABELS = {
    "N": "Narrative (what is depicted, scene and story)",
    "F": "Formal (composition, color, technique, brushwork)",
    "E": "Emotional (mood, atmosphere, affective tone)",
    "H": "Historical (art-historical context, movement, cultural significance)",
}

# Cleaned consolidated dataset (single source of truth)
DATASET_FILE = "output/polyart_dataset_clean_full4.json"

ALL_CONDITIONS = [
    "N", "F", "E", "H",          # singles
    "NFE", "NFH", "NEH", "FEH",  # leave-one-out
    "NFEH",                       # full
]

SINGLE_CONDITIONS = {"N", "F", "E", "H"}

# Synthesis prompt — used for all multi-perspective conditions
SYNTHESIS_PROMPT = """\
You are an art description writer.

The following are {n} interpretive perspectives on the same painting, \
"{title}" by {artist}:

{block}

Write a single coherent ~80-word description that integrates all these \
perspectives into one unified passage. Do not label or list the perspectives \
separately. Write fluently in third person. Output only the description, \
nothing else."""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_indices(repo_root: Path) -> dict:
    """
    Load polyart_dataset_clean_full4.json and build per-perspective indices:
      key -> {image_id: {"caption": str, "title": str, "artist": str, "style": str, "date": str}}
    """
    import json
    dataset_path = repo_root / DATASET_FILE
    print(f"Loading {dataset_path} ...")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    indices = {k: {} for k in PERSPECTIVE_FULL}
    for r in data:
        iid = r.get("image_id", "")
        if not iid:
            continue
        meta = {
            "title":  r.get("title", ""),
            "artist": r.get("artist", ""),
            "style":  r.get("style", ""),
            "date":   str(r.get("date", "")),
        }
        for key, field in PERSPECTIVE_FULL.items():
            cap = r.get(field)
            if cap:
                indices[key][iid] = {"caption": cap, **meta}

    for k, idx in indices.items():
        print(f"  {k}: {len(idx):,} records")
    return indices


# ---------------------------------------------------------------------------
# Sample selection — stratified by style, saved for reuse
# ---------------------------------------------------------------------------

def select_sample(indices: dict, n: int, seed: int) -> pd.DataFrame:
    """Pick n paintings present in ALL 4 perspectives, stratified by style."""
    common = set(indices["N"])
    for k in ["F", "E", "H"]:
        common &= set(indices[k])
    print(f"Paintings with all 4 perspectives: {len(common):,}")

    rows = [
        {
            "image_id": iid,
            **{f: indices["N"][iid][f] for f in ("title", "artist", "style", "date")},
        }
        for iid in common
    ]
    df = pd.DataFrame(rows)

    if n >= len(df):
        print(f"  n_sample={n} >= pool={len(df)}; using all.")
        return df.reset_index(drop=True)

    n_styles = df["style"].nunique()
    per_style = max(1, n // n_styles)

    sampled = (
        df.groupby("style", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_style), random_state=seed))
    )
    if len(sampled) < n:
        rest = df[~df["image_id"].isin(sampled["image_id"])]
        extra = rest.sample(min(n - len(sampled), len(rest)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.head(n).reset_index(drop=True)
    print(f"  Selected {len(sampled)} paintings across {sampled['style'].nunique()} styles")
    return sampled


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_existing(path: str) -> tuple[list, set]:
    results = []
    if os.path.exists(path):
        try:
            results = pd.read_json(path).to_dict("records")
            print(f"  Resumed: {len(results)} records from {path}")
        except Exception as e:
            print(f"  Warning: could not resume ({e}); starting fresh")
    return results, {r["image_id"] for r in results}


def save(results: list, path: str) -> None:
    pd.DataFrame(results).to_json(path, orient="records", indent=2)


# ---------------------------------------------------------------------------
# Singles — just copy existing caption
# ---------------------------------------------------------------------------

def run_single(condition: str, sample_df: pd.DataFrame, indices: dict, out_path: str) -> None:
    results, processed = load_existing(out_path)
    idx = indices[condition]

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"{condition} (copy)"):
        iid = row["image_id"]
        if iid in processed or iid not in idx:
            continue
        results.append({
            "image_id":            iid,
            "title":               row["title"],
            "artist":              row["artist"],
            "style":               row["style"],
            "date":                row["date"],
            "condition":           condition,
            "perspectives_used":   [PERSPECTIVE_FULL[condition]],
            "synthesized_caption": idx[iid]["caption"],
        })
        processed.add(iid)

    save(results, out_path)
    print(f"  {len(results)} records → {out_path}")


# ---------------------------------------------------------------------------
# Multi — synthesize with Qwen3-8B via vLLM (model loaded once externally)
# ---------------------------------------------------------------------------

def build_prompt(condition: str, iid: str, row: pd.Series, indices: dict) -> str:
    block = "\n\n".join(
        f"- {PERSPECTIVE_LABELS[k]}:\n  {indices[k][iid]['caption']}"
        for k in condition
    )
    return SYNTHESIS_PROMPT.format(
        n=len(condition),
        title=row["title"],
        artist=row["artist"],
        block=block,
    )


def run_multi(
    condition: str,
    sample_df: pd.DataFrame,
    indices: dict,
    out_path: str,
    llm,
    tokenizer,
    sampling_params,
    batch_size: int = 512,
) -> None:
    results, processed = load_existing(out_path)

    todo = sample_df[~sample_df["image_id"].isin(processed)].reset_index(drop=True)
    if todo.empty:
        print(f"  {condition}: already complete.")
        return

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start : start + batch_size]

        prompts, valid_rows = [], []
        for _, row in batch.iterrows():
            iid = row["image_id"]
            if not all(iid in indices[k] for k in condition):
                print(f"  Skipping {iid}: missing perspective(s)")
                continue

            messages = [
                {"role": "system", "content": "You are a helpful art description writer."},
                {"role": "user",   "content": build_prompt(condition, iid, row, indices)},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # disable Qwen3 chain-of-thought
            )
            prompts.append(text)
            valid_rows.append(row)

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params)

        for row, out in zip(valid_rows, outputs):
            results.append({
                "image_id":            row["image_id"],
                "title":               row["title"],
                "artist":              row["artist"],
                "style":               row["style"],
                "date":                row["date"],
                "condition":           condition,
                "perspectives_used":   [PERSPECTIVE_FULL[k] for k in condition],
                "synthesized_caption": out.outputs[0].text.strip(),
            })
            processed.add(row["image_id"])

        save(results, out_path)
        print(f"  [{condition}] checkpoint: {len(results)}/{len(todo)}")

    print(f"  [{condition}] done. {len(results)} records → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: synthesize perspective prompts for image regeneration experiment"
    )
    parser.add_argument(
        "--condition", default="all",
        help=f"Condition to run ({', '.join(ALL_CONDITIONS)}) or 'all' (default: all)"
    )
    parser.add_argument("--n_sample",   type=int,   default=1000,
                        help="Paintings to sample; 50 per style × 20 styles = 1000 (default: 1000)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--model_name", type=str,   default="Qwen/Qwen3-8B",
                        help="vLLM model for synthesis (default: Qwen/Qwen3-8B)")
    parser.add_argument("--batch_size", type=int,   default=512,
                        help="vLLM prompt batch size (default: 512)")
    parser.add_argument("--output_dir", type=str,   default=None,
                        help="Override output directory")
    parser.add_argument("--gpu_mem",    type=float, default=0.85,
                        help="vLLM gpu_memory_utilization (default: 0.85)")
    args = parser.parse_args()

    repo_root  = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "output" / "synthesis_prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve conditions
    if args.condition == "all":
        conditions = ALL_CONDITIONS
    elif args.condition in ALL_CONDITIONS:
        conditions = [args.condition]
    else:
        parser.error(f"Unknown condition '{args.condition}'. Options: {ALL_CONDITIONS + ['all']}")

    # Load perspective data once
    indices = load_all_indices(repo_root)

    # Load or create sample (filename reflects actual size for reproducibility)
    sample_csv = output_dir / f"sample_{args.n_sample}.csv"
    if sample_csv.exists():
        print(f"Loading existing sample from {sample_csv}")
        sample_df = pd.read_csv(sample_csv)
    else:
        sample_df = select_sample(indices, args.n_sample, args.seed)
        sample_df.to_csv(sample_csv, index=False)
        print(f"Saved sample → {sample_csv}")

    single_conds = [c for c in conditions if c in SINGLE_CONDITIONS]
    multi_conds  = [c for c in conditions if c not in SINGLE_CONDITIONS]

    # --- Singles (no GPU needed) ---
    for cond in single_conds:
        print(f"\n=== {cond} (single — copy) ===")
        cond_dir = output_dir / cond
        cond_dir.mkdir(exist_ok=True)
        run_single(cond, sample_df, indices, str(cond_dir / f"synthesis_{cond}.json"))

    # --- Multi (load vLLM once, run all conditions) ---
    if multi_conds:
        print(f"\nLoading vLLM model: {args.model_name} ...")
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
            max_tokens=200,
            repetition_penalty=1.05,
        )

        for cond in multi_conds:
            print(f"\n=== {cond} (synthesize) ===")
            cond_dir = output_dir / cond
            cond_dir.mkdir(exist_ok=True)
            run_multi(
                cond, sample_df, indices,
                str(cond_dir / f"synthesis_{cond}.json"),
                llm, tokenizer, sampling_params,
                args.batch_size,
            )

    print(f"\nPhase 1 complete. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
