"""
phase3_eval.py

Evaluates perspective complementarity by measuring reconstruction fidelity
between regenerated images (Phase 2 output) and the original WikiArt paintings.

Three complementary metrics are computed for each (painting, condition, model) triple:

  1. CLIP cosine similarity  — semantic / style fidelity
     openai/clip-vit-large-patch14
     Measures how well the regenerated image matches the style and semantic
     character of the original (CLIP space is organized by art-language concepts).

  2. DINOv3 cosine similarity — compositional / structural fidelity
     facebook/dinov3-vitl16-pretrain-lvd1689m
     Self-supervised patch features capture spatial layout and structure
     independent of style labels.

  3. Emotion agreement       — affective fidelity
     CLIP zero-shot against 9 ARTEMIS emotion labels.
     Top-1 emotion of regenerated image vs. original: 1 if match, 0 otherwise.

The relative pattern across conditions is the finding — not absolute scores.
Expected: NFEH > any leave-one-out > best single, with each perspective
contributing most to the metric aligned with its type.

Output
------
  output/phase3_results/
    results_raw.json          ← per-painting per-condition per-model scores
    results_summary.json      ← mean ± std per condition × metric × model
    results_table.csv         ← paper-ready table (conditions as rows)

Usage
-----
  python scripts/phase3_eval.py
  python scripts/phase3_eval.py --batch_size 64

SLURM
-----
  bash slurm/run_eval.sh
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoImageProcessor, AutoModel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
REGEN_DIR   = REPO_ROOT / "output" / "regenerated_images"
PROMPTS_DIR = REPO_ROOT / "output" / "synthesis_prompts"
OUT_DIR     = REPO_ROOT / "output" / "phase3_results"
WIKIART_DIR = Path("/gpfs/work5/0/prjs0996/data/wikiart/Images")

ALL_CONDITIONS = ["N", "F", "E", "H", "NFE", "NFH", "NEH", "FEH", "NFEH", "U"]
ALL_MODELS     = ["flux2_klein", "qwen_image"]
EVAL_SIZE      = 512

# ARTEMIS emotion vocabulary for zero-shot classification
EMOTION_LABELS = [
    "amusement", "awe", "contentment", "excitement",
    "fear", "sadness", "disgust", "anger", "something else",
]
EMOTION_PROMPTS = [f"a painting that evokes {e}" for e in EMOTION_LABELS]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: Path, size: int = EVAL_SIZE) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img.copy()


def safe_to_original(safe_id: str) -> Path:
    """Convert safe filename (__ separator) back to WikiArt relative path."""
    rel = safe_id.replace("__", "/", 1)   # only first __ is the style/name split
    return WIKIART_DIR / rel


# ---------------------------------------------------------------------------
# Metric models
# ---------------------------------------------------------------------------

class CLIPEvaluator:
    def __init__(self, device: str):
        model_id = "openai/clip-vit-large-patch14"
        print(f"  Loading CLIP ({model_id}) ...")
        self.device    = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model     = CLIPModel.from_pretrained(model_id).to(device).eval()

        # Pre-encode emotion text prompts (done once)
        text_inputs = self.processor(
            text=EMOTION_PROMPTS,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            self.emotion_text_feats = F.normalize(
                self.model.get_text_features(**text_inputs), dim=-1
            )  # (9, D)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats  = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)  # (B, D)

    @torch.no_grad()
    def clip_similarity(
        self,
        orig_feats: torch.Tensor,
        regen_feats: torch.Tensor,
    ) -> list[float]:
        """Cosine similarity between original and regenerated image features."""
        sims = (orig_feats * regen_feats).sum(dim=-1)  # (B,)
        return sims.cpu().tolist()

    @torch.no_grad()
    def emotion_agreement(
        self,
        orig_feats: torch.Tensor,
        regen_feats: torch.Tensor,
    ) -> list[float]:
        """1 if top-1 emotion label matches between original and regenerated, else 0."""
        orig_logits  = orig_feats  @ self.emotion_text_feats.T   # (B, 9)
        regen_logits = regen_feats @ self.emotion_text_feats.T   # (B, 9)
        orig_top1    = orig_logits.argmax(dim=-1)
        regen_top1   = regen_logits.argmax(dim=-1)
        return (orig_top1 == regen_top1).float().cpu().tolist()


class DINOEvaluator:
    def __init__(self, device: str):
        model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        print(f"  Loading DINOv3 ({model_id}) ...")
        self.device    = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model     = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out    = self.model(**inputs)
        feats  = out.last_hidden_state[:, 0, :]   # CLS token
        return F.normalize(feats, dim=-1)          # (B, D)

    @torch.no_grad()
    def dino_similarity(
        self,
        orig_feats: torch.Tensor,
        regen_feats: torch.Tensor,
    ) -> list[float]:
        sims = (orig_feats * regen_feats).sum(dim=-1)
        return sims.cpu().tolist()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(batch_size: int, device: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load painting list from sample ---
    sample_csv = PROMPTS_DIR / "sample_1000.csv"
    import pandas as pd
    sample_df = pd.read_csv(sample_csv)
    image_ids = sample_df["image_id"].tolist()   # e.g. "Abstract_Expressionism/aaron-siskind_..."
    styles    = sample_df["style"].tolist()
    print(f"Evaluating {len(image_ids)} paintings × {len(ALL_CONDITIONS)} conditions "
          f"× {len(ALL_MODELS)} models = "
          f"{len(image_ids)*len(ALL_CONDITIONS)*len(ALL_MODELS):,} image pairs\n")

    # --- Checkpoint: load existing results ---
    raw_path = OUT_DIR / "results_raw.json"
    if raw_path.exists():
        raw_results = json.load(open(raw_path))
        print(f"Resuming from checkpoint: {len(raw_results)} paintings already done")
    else:
        raw_results = {}

    done_ids = set(raw_results.keys())

    # --- Load metric models ---
    print("Loading metric models ...")
    clip_eval = CLIPEvaluator(device)
    dino_eval = DINOEvaluator(device)
    print("  All metric models loaded.\n")

    # Build safe_id → image_id mapping
    def to_safe(iid: str) -> str:
        parts = iid.split("/", 1)
        return "__".join(parts)

    # --- Iterate paintings in batches ---
    remaining = [iid for iid in image_ids if iid not in done_ids]

    for batch_start in tqdm(range(0, len(remaining), batch_size),
                            desc="Evaluating batches"):
        batch_ids    = remaining[batch_start : batch_start + batch_size]
        batch_styles = {iid: styles[image_ids.index(iid)] for iid in batch_ids}

        # Load original images once per batch
        orig_images = {}
        for iid in batch_ids:
            orig_path = WIKIART_DIR / iid
            if not orig_path.exists():
                print(f"  Warning: original not found — {orig_path}")
                continue
            orig_images[iid] = load_image(orig_path)

        if not orig_images:
            continue

        valid_ids = list(orig_images.keys())

        # Encode originals once
        orig_imgs_list = [orig_images[iid] for iid in valid_ids]
        orig_clip = clip_eval.encode_images(orig_imgs_list)
        orig_dino = dino_eval.encode_images(orig_imgs_list)

        # Evaluate each condition × model
        batch_results = {iid: {"style": batch_styles[iid], "conditions": {}}
                         for iid in valid_ids}

        for cond in ALL_CONDITIONS:
            for model_name in ALL_MODELS:
                regen_imgs = []
                valid_for_pair = []

                for iid in valid_ids:
                    safe_id   = to_safe(iid)
                    regen_path = REGEN_DIR / model_name / cond / safe_id
                    if not regen_path.exists():
                        continue
                    regen_imgs.append(load_image(regen_path))
                    valid_for_pair.append(iid)

                if not regen_imgs:
                    continue

                # Get matching original features
                idxs = [valid_ids.index(iid) for iid in valid_for_pair]
                pair_orig_clip = orig_clip[idxs]
                pair_orig_dino = orig_dino[idxs]

                regen_clip  = clip_eval.encode_images(regen_imgs)
                regen_dino  = dino_eval.encode_images(regen_imgs)

                clip_sims    = clip_eval.clip_similarity(pair_orig_clip, regen_clip)
                dino_sims    = dino_eval.dino_similarity(pair_orig_dino, regen_dino)
                emotion_agr  = clip_eval.emotion_agreement(pair_orig_clip, regen_clip)

                for i, iid in enumerate(valid_for_pair):
                    if cond not in batch_results[iid]["conditions"]:
                        batch_results[iid]["conditions"][cond] = {}
                    batch_results[iid]["conditions"][cond][model_name] = {
                        "clip_sim":       round(clip_sims[i],   4),
                        "dino_sim":       round(dino_sims[i],   4),
                        "emotion_agree":  round(emotion_agr[i], 4),
                    }

        raw_results.update(batch_results)

        # Checkpoint after every batch
        with open(raw_path, "w") as f:
            json.dump(raw_results, f, indent=2)

    print(f"\nRaw results saved → {raw_path}")
    return raw_results


# ---------------------------------------------------------------------------
# Aggregate results → summary table
# ---------------------------------------------------------------------------

def summarize(raw_results: dict):
    metrics = ["clip_sim", "dino_sim", "emotion_agree"]

    # summary[condition][model][metric] = [scores...]
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for iid, rec in raw_results.items():
        for cond, models in rec["conditions"].items():
            for model_name, scores in models.items():
                for m in metrics:
                    if m in scores:
                        summary[cond][model_name][m].append(scores[m])

    # Build summary dict
    summary_out = {}
    for cond in ALL_CONDITIONS:
        summary_out[cond] = {}
        for model_name in ALL_MODELS:
            summary_out[cond][model_name] = {}
            for m in metrics:
                vals = summary[cond][model_name][m]
                if vals:
                    summary_out[cond][model_name][m] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "std":  round(float(np.std(vals)),  4),
                        "n":    len(vals),
                    }

    summary_path = OUT_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Summary saved → {summary_path}")

    # Build CSV table (paper-ready)
    import csv
    csv_path = OUT_DIR / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["condition"]
        for model_name in ALL_MODELS:
            for m in metrics:
                header.append(f"{model_name}/{m}")
        writer.writerow(header)

        for cond in ALL_CONDITIONS:
            row = [cond]
            for model_name in ALL_MODELS:
                for m in metrics:
                    entry = summary_out.get(cond, {}).get(model_name, {}).get(m, {})
                    if entry:
                        row.append(f"{entry['mean']:.4f}±{entry['std']:.4f}")
                    else:
                        row.append("—")
            writer.writerow(row)
    print(f"Table saved  → {csv_path}")

    # Pretty-print to terminal
    print("\n=== Results Table (mean ± std) ===\n")
    print(f"{'Condition':<10}", end="")
    for model_name in ALL_MODELS:
        for m in ["clip_sim", "dino_sim", "emotion_agree"]:
            short = f"{model_name[:5]}/{m[:4]}"
            print(f"  {short:>14}", end="")
    print()

    for cond in ALL_CONDITIONS:
        print(f"{cond:<10}", end="")
        for model_name in ALL_MODELS:
            for m in metrics:
                entry = summary_out.get(cond, {}).get(model_name, {}).get(m, {})
                if entry:
                    print(f"  {entry['mean']:>7.4f}±{entry['std']:.3f}", end="")
                else:
                    print(f"  {'—':>14}", end="")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: fidelity evaluation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Images per batch (default: 64)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start from scratch")
    args = parser.parse_args()

    if args.fresh:
        raw_path = OUT_DIR / "results_raw.json"
        if raw_path.exists():
            raw_path.unlink()
            print("Deleted old results_raw.json (--fresh flag set)")

    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}\n")

    raw_results = run_eval(args.batch_size, args.device)
    summarize(raw_results)
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
