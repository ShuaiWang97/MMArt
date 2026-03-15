"""
clean_dataset.py

Cleans polyart_dataset.json in four targeted steps and produces a
publication-ready version alongside a detailed cleaning report.

Cleaning steps
--------------
1. Null Chinese-contaminated captions   — any caption containing CJK characters
2. Null very-short captions             — fewer than MIN_WORDS words (likely errors)
3. Truncate overlong e_formal captions  — at last sentence boundary ≤ MAX_FORMAL_WORDS
4. Null duplicate e_formal captions     — exact duplicates = generic fallback responses

Two-tier output
---------------
  polyart_dataset_clean.json/jsonl/parquet
      Full 75,336 records; affected perspectives set to null (not deleted).
      Honest representation — dataset users see exactly what's available.

  polyart_dataset_clean_full4.json/jsonl/parquet
      Subset containing only paintings where all 4 perspectives are non-null.
      Used as the experiment pool for phase1_synthesize.py (image regeneration).

  cleaning_report.json
      Per-step counts, affected image_ids, and final coverage stats.

Usage
-----
  python scripts/clean_dataset.py
  python scripts/clean_dataset.py --input output/polyart_dataset.json
  python scripts/clean_dataset.py --input output/polyart_dataset.json \\
                                   --output_dir output/release
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSPECTIVES = ["e_narrative", "e_formal", "e_emotional", "e_historical"]

# CJK Unified Ideographs (Chinese / Japanese / Korean)
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]")

# Caption shorter than this is considered an error
MIN_WORDS = 20

# e_formal captions longer than this get truncated at sentence boundary
MAX_FORMAL_WORDS = 150

# Sentence-ending punctuation for truncation
SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def word_count(text: str) -> int:
    return len(text.split())


def truncate_at_sentence(text: str, max_words: int) -> str:
    """
    Return the longest prefix of `text` that ends on a sentence boundary
    and is at most `max_words` words.  Falls back to a hard word-level cut
    if no sentence boundary is found within the limit.
    """
    sentences = SENT_END_RE.split(text)
    result = []
    total = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if total + sent_words > max_words:
            break
        result.append(sent)
        total += sent_words
    if result:
        return " ".join(result)
    # No sentence fits — hard word-level truncation
    return " ".join(text.split()[:max_words])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean(data: list[dict], report: dict) -> list[dict]:

    # -----------------------------------------------------------------------
    # Step 1 — Null CJK-contaminated captions
    # -----------------------------------------------------------------------
    step1 = defaultdict(list)
    for rec in tqdm(data, desc="Step 1: CJK check"):
        for p in PERSPECTIVES:
            cap = rec.get(p)
            if cap and has_cjk(cap):
                step1[p].append(rec["image_id"])
                rec[p] = None

    report["step1_cjk"] = {
        p: {"count": len(ids), "image_ids": ids}
        for p, ids in step1.items()
    }
    total1 = sum(len(v) for v in step1.values())
    print(f"  Step 1 — nulled {total1} CJK-contaminated captions "
          f"({', '.join(f'{p}:{len(step1[p])}' for p in PERSPECTIVES)})")

    # -----------------------------------------------------------------------
    # Step 2 — Null very-short captions
    # -----------------------------------------------------------------------
    step2 = defaultdict(list)
    for rec in tqdm(data, desc="Step 2: short caption check"):
        for p in PERSPECTIVES:
            cap = rec.get(p)
            if cap and word_count(cap) < MIN_WORDS:
                step2[p].append(rec["image_id"])
                rec[p] = None

    report["step2_short"] = {
        "min_words_threshold": MIN_WORDS,
        **{p: {"count": len(ids), "image_ids": ids} for p, ids in step2.items()},
    }
    total2 = sum(len(v) for v in step2.values())
    print(f"  Step 2 — nulled {total2} short captions (<{MIN_WORDS} words) "
          f"({', '.join(f'{p}:{len(step2[p])}' for p in PERSPECTIVES)})")

    # -----------------------------------------------------------------------
    # Step 3 — Truncate overlong e_formal captions
    # -----------------------------------------------------------------------
    step3_ids = []
    step3_before = []
    step3_after  = []
    for rec in tqdm(data, desc="Step 3: truncate e_formal"):
        cap = rec.get("e_formal")
        if cap and word_count(cap) > MAX_FORMAL_WORDS:
            truncated = truncate_at_sentence(cap, MAX_FORMAL_WORDS)
            step3_ids.append(rec["image_id"])
            step3_before.append(word_count(cap))
            step3_after.append(word_count(truncated))
            rec["e_formal"] = truncated

    report["step3_truncate_formal"] = {
        "max_words_threshold": MAX_FORMAL_WORDS,
        "count": len(step3_ids),
        "avg_words_before": round(sum(step3_before) / max(len(step3_before), 1), 1),
        "avg_words_after":  round(sum(step3_after)  / max(len(step3_after),  1), 1),
        "image_ids": step3_ids,
    }
    print(f"  Step 3 — truncated {len(step3_ids)} overlong e_formal captions "
          f"(avg {report['step3_truncate_formal']['avg_words_before']}w → "
          f"{report['step3_truncate_formal']['avg_words_after']}w)")

    # -----------------------------------------------------------------------
    # Step 4 — Null duplicate e_formal captions
    # -----------------------------------------------------------------------
    seen_formal: dict[str, str] = {}      # caption text → first image_id
    step4_ids = []
    for rec in tqdm(data, desc="Step 4: duplicate e_formal check"):
        cap = rec.get("e_formal")
        if not cap:
            continue
        if cap in seen_formal:
            step4_ids.append({
                "image_id":   rec["image_id"],
                "first_seen": seen_formal[cap],
            })
            rec["e_formal"] = None
        else:
            seen_formal[cap] = rec["image_id"]

    report["step4_duplicates_formal"] = {
        "count": len(step4_ids),
        "affected": step4_ids,
    }
    print(f"  Step 4 — nulled {len(step4_ids)} duplicate e_formal captions")

    # -----------------------------------------------------------------------
    # Recompute n_perspectives
    # -----------------------------------------------------------------------
    for rec in data:
        rec["n_perspectives"] = sum(1 for p in PERSPECTIVES if rec.get(p))

    return data


def compute_coverage(data: list[dict]) -> dict:
    total = len(data)
    coverage = {}
    for p in PERSPECTIVES:
        n = sum(1 for r in data if r.get(p))
        coverage[p] = {"count": n, "pct": round(n / total * 100, 2)}
    all4 = sum(1 for r in data if all(r.get(p) for p in PERSPECTIVES))
    coverage["all_4"] = {"count": all4, "pct": round(all4 / total * 100, 2)}
    return coverage


def save_outputs(data: list[dict], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (pretty, for inspection)
    json_path = out_dir / f"polyart_dataset_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)")

    # JSONL (streaming-friendly)
    jsonl_path = out_dir / f"polyart_dataset_{tag}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {jsonl_path}  ({jsonl_path.stat().st_size / 1e6:.1f} MB)")

    # Parquet (HuggingFace-preferred)
    parquet_path = out_dir / f"polyart_dataset_{tag}.parquet"
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path}  ({parquet_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Clean PolyArt dataset")
    parser.add_argument("--input", default="output/polyart_dataset.json",
                        help="Path to polyart_dataset.json")
    parser.add_argument("--output_dir", default="output",
                        help="Directory for cleaned outputs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = repo_root / args.input
    out_dir    = repo_root / args.output_dir

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    print(f"Loading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data):,} records loaded\n")

    report: dict = {
        "input_file": str(input_path),
        "total_records": len(data),
        "coverage_before": compute_coverage(data),
    }

    # -----------------------------------------------------------------------
    # Clean
    # -----------------------------------------------------------------------
    print("Cleaning ...")
    data = clean(data, report)
    print()

    report["coverage_after"] = compute_coverage(data)

    # -----------------------------------------------------------------------
    # Tier 1 — full dataset (some perspectives may be null)
    # -----------------------------------------------------------------------
    print("Saving tier-1 (full dataset) ...")
    save_outputs(data, out_dir, tag="clean")
    print()

    # -----------------------------------------------------------------------
    # Tier 2 — experiment subset (all 4 perspectives non-null)
    # -----------------------------------------------------------------------
    full4 = [r for r in data if r.get("n_perspectives") == 4]
    report["full4_count"] = len(full4)
    print(f"Saving tier-2 (all-4-perspectives subset: {len(full4):,} records) ...")
    save_outputs(full4, out_dir, tag="clean_full4")
    print()

    # -----------------------------------------------------------------------
    # Cleaning report
    # -----------------------------------------------------------------------
    report_path = out_dir / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Cleaning report saved → {report_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Total records         : {report['total_records']:,}")
    print(f"  All-4 before cleaning : {report['coverage_before']['all_4']['count']:,}  "
          f"({report['coverage_before']['all_4']['pct']}%)")
    print(f"  All-4 after cleaning  : {report['coverage_after']['all_4']['count']:,}  "
          f"({report['coverage_after']['all_4']['pct']}%)")
    print(f"  Experiment pool (tier-2): {len(full4):,} paintings")
    print("\nPer-perspective coverage after cleaning:")
    for p in PERSPECTIVES:
        c = report["coverage_after"][p]
        print(f"  {p:15s}: {c['count']:,}  ({c['pct']}%)")


if __name__ == "__main__":
    main()
