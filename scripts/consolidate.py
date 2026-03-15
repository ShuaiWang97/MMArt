"""
consolidate.py

Merges all four perspective outputs (32 job files) into a single unified
dataset with one record per painting.

Outputs:
  output/polyart_dataset.parquet   ← HuggingFace-ready (preferred format)
  output/polyart_dataset.jsonl     ← streaming-friendly, one JSON per line
  output/polyart_dataset_stats.json ← coverage / quality summary

Schema per record:
  image_id          str   — relative path used as unique key
  title             str
  artist            str
  style             str   — WikiArt style label
  date              str
  e_narrative       str | null
  e_formal          str | null
  e_emotional       str | null
  e_historical      str | null
  dominant_emotion  str | null   — majority-vote from ARTEMIS-v2 (emotional only)
  artemis_coverage  bool| null   — True if ARTEMIS utterances were available
  rag_sim           float| null  — cosine sim of best RAG chunk (historical only)
  n_perspectives    int   — number of non-null perspectives (0–4)

Usage:
  python scripts/consolidate.py
  python scripts/consolidate.py --output_dir output/release
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Source file locations
# ---------------------------------------------------------------------------

SOURCES = {
    "e_narrative": {
        "dir":    "output/e_narrative",
        "prefix": "narrative_job",
        "n_jobs": 8,
        "caption_field": "generated_caption",
        "extra_fields":  [],
    },
    "e_formal": {
        "dir":    "output/formal_analysis",
        "prefix": "llava_captions_job",
        "n_jobs": 8,
        "caption_field": "generated_caption",
        "extra_fields":  [],
    },
    "e_emotional": {
        "dir":    "output/e_emotional",
        "prefix": "emotional_job",
        "n_jobs": 8,
        "caption_field": "generated_caption",
        "extra_fields":  ["dominant_emotion", "artemis_coverage"],
    },
    "e_historical": {
        "dir":    "output/e_historical",
        "prefix": "historical_job",
        "n_jobs": 8,
        "caption_field": "generated_caption",
        "extra_fields":  ["rag_sim"],
    },
}

# Metadata fields — taken from whichever perspective has them
META_FIELDS = ["title", "artist", "style", "date"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_perspective(repo_root: Path, key: str) -> dict:
    """
    Load all job files for one perspective.
    Returns: image_id -> {caption, dominant_emotion?, artemis_coverage?, rag_sim?}
    """
    cfg   = SOURCES[key]
    index = {}
    missing = 0

    for job_id in range(cfg["n_jobs"]):
        fpath = repo_root / cfg["dir"] / f"{cfg['prefix']}_{job_id}.json"
        if not fpath.exists():
            print(f"  Warning: {fpath} not found")
            missing += 1
            continue

        records = pd.read_json(fpath, dtype={"date": str}).to_dict("records")
        for r in records:
            iid = r.get("image_id", "")
            cap = r.get(cfg["caption_field"], "")
            if not iid or not cap:
                continue

            entry = {"caption": cap}
            # Pull metadata (used to build unified records)
            for f in META_FIELDS:
                entry[f] = r.get(f, "")
            # Pull perspective-specific extras
            for f in cfg["extra_fields"]:
                if f in r:
                    entry[f] = r[f]

            index[iid] = entry

    print(f"  {key}: {len(index):,} records ({missing} job files missing)")
    return index


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_dataset(repo_root: Path) -> pd.DataFrame:
    print("Loading perspectives ...")
    indices = {k: load_perspective(repo_root, k) for k in SOURCES}

    # Union of all image_ids
    all_ids = set()
    for idx in indices.values():
        all_ids |= set(idx)
    print(f"\nTotal unique paintings across all perspectives: {len(all_ids):,}")

    rows = []
    for iid in tqdm(sorted(all_ids), desc="Merging"):
        # Collect metadata from whichever perspective has this painting
        meta = {}
        for key in SOURCES:
            if iid in indices[key]:
                for f in META_FIELDS:
                    if not meta.get(f):
                        meta[f] = indices[key][iid].get(f, "")

        row = {
            "image_id": iid,
            "title":    meta.get("title",  ""),
            "artist":   meta.get("artist", ""),
            "style":    meta.get("style",  ""),
            "date":     meta.get("date",   ""),
        }

        # Four perspective captions
        for key in SOURCES:
            entry = indices[key].get(iid)
            row[key] = entry["caption"] if entry else None

        # Perspective-specific extras (null if not available)
        e_entry = indices["e_emotional"].get(iid, {})
        row["dominant_emotion"] = e_entry.get("dominant_emotion", None)
        row["artemis_coverage"] = e_entry.get("artemis_coverage", None)

        h_entry = indices["e_historical"].get(iid, {})
        row["rag_sim"] = h_entry.get("rag_sim", None)

        # Convenience count
        row["n_perspectives"] = sum(
            1 for k in SOURCES if row[k] is not None
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce column order
    col_order = [
        "image_id", "title", "artist", "style", "date",
        "e_narrative", "e_formal", "e_emotional", "e_historical",
        "dominant_emotion", "artemis_coverage", "rag_sim",
        "n_perspectives",
    ]
    df = df[col_order]
    return df


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    stats = {
        "total_paintings": total,
        "perspective_coverage": {
            key: {
                "count": int(df[key].notna().sum()),
                "pct":   round(df[key].notna().sum() / total * 100, 2),
            }
            for key in SOURCES
        },
        "all_4_perspectives": int((df["n_perspectives"] == 4).sum()),
        "at_least_3":         int((df["n_perspectives"] >= 3).sum()),
        "style_distribution": df["style"].value_counts().head(10).to_dict(),
        "artemis_grounded":   int(df["artemis_coverage"].eq(True).sum()),
        "rag_grounded":       int(df["rag_sim"].notna().sum()),
    }
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consolidate PolyArt perspective outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <repo_root>/output)")
    args = parser.parse_args()

    repo_root  = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(repo_root)

    # --- JSON (primary, human-readable) ---
    json_path = output_dir / "polyart_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
    print(f"\nSaved json     → {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)")

    # --- JSONL (streaming-friendly, good for large-scale loading) ---
    jsonl_path = output_dir / "polyart_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved jsonl    → {jsonl_path}  ({jsonl_path.stat().st_size / 1e6:.1f} MB)")

    # --- Parquet (HuggingFace upload) ---
    parquet_path = output_dir / "polyart_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet  → {parquet_path}  ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # --- Stats ---
    stats = compute_stats(df)
    stats_path = output_dir / "polyart_dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats    → {stats_path}")

    # Print summary
    print("\n=== PolyArt Dataset Summary ===")
    print(f"Total paintings  : {stats['total_paintings']:,}")
    print(f"All 4 perspectives: {stats['all_4_perspectives']:,}")
    print(f"At least 3        : {stats['at_least_3']:,}")
    for key, v in stats["perspective_coverage"].items():
        print(f"  {key:15s}: {v['count']:,}  ({v['pct']}%)")
    print(f"ARTEMIS grounded  : {stats['artemis_grounded']:,}")
    print(f"RAG grounded      : {stats['rag_grounded']:,}")


if __name__ == "__main__":
    main()
