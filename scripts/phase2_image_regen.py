"""
phase2_image_regen.py

Regenerates images from synthesized text prompts (Phase 1 output) using
a text-to-image model. Supports two models:

  flux2_klein    — black-forest-labs/FLUX.2-klein-4B  (Flux2KleinPipeline, 4 steps, distilled)
  qwen_image     — Qwen/Qwen-Image-2512               (DiffusionPipeline, 25 steps)

Resolution strategy
-------------------
  Both models generate at 1024×1024 (their optimal range) then
  downsample to 512×512 for evaluation — consistent eval resolution.

This ensures Phase 3 evaluation always compares at a consistent 512×512,
while each model runs at a sensible resolution.

For each of the 9 experimental conditions (N, F, E, H, NFE, NFH, NEH, FEH, NFEH)
and each painting in the 1,000-painting sample, one image is generated from the
synthesized caption. The original WikiArt image is the reconstruction target in
Phase 3 evaluation.

Output layout
-------------
  output/regenerated_images/<model>/<condition>/<safe_image_id>.jpg

where <safe_image_id> replaces "/" with "__" to be filesystem-safe.

Usage
-----
  python scripts/phase2_image_regen.py --model flux2_klein --condition N
  python scripts/phase2_image_regen.py --model qwen_image   --condition NFEH
  python scripts/phase2_image_regen.py --model flux2_klein  --condition all

Model download (first run only)
--------------------------------
  Models are downloaded automatically from HuggingFace to:
    /gpfs/work5/0/prjs0996/.cache/huggingface/hub/
  FLUX.1-schnell: ~12 GB   Qwen-Image-2512: ~varies

Checkpoint/recovery
-------------------
  Already-generated images are skipped on re-run — safe to interrupt and resume.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONDITIONS = ["N", "F", "E", "H", "NFE", "NFH", "NEH", "FEH", "NFEH", "U"]

# Eval resolution — all saved images are resized to this for Phase 3
EVAL_SIZE = 512

MODEL_CONFIGS = {
    "flux2_klein": {
        "model_id":       "black-forest-labs/FLUX.2-klein-4B",
        "default_steps":  4,          # distilled model — 4 steps is optimal
        "guidance_scale": 1.0,        # distilled, minimal CFG
        "gen_height":     1024,       # optimal range; downsampled to 512 after
        "gen_width":      1024,
        "supports_negative": False,
    },
    "qwen_image": {
        "model_id":       "Qwen/Qwen-Image-2512",
        "default_steps":  25,
        "true_cfg_scale": 4.0,
        "gen_height":     1024,       # optimal range; downsampled to 512 after
        "gen_width":      1024,
        "supports_negative": True,
        "negative_prompt": (
            "blurry, low quality, deformed, ugly, text, watermark, "
            "signature, extra limbs, bad anatomy"
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(image_id: str) -> str:
    """Convert 'Style/artist_title.jpg' → 'Style__artist_title.jpg'."""
    return image_id.replace("/", "__")


def load_pipeline(model_key: str):
    """Load and return the diffusion pipeline for the given model key."""
    cfg = MODEL_CONFIGS[model_key]
    dtype = torch.bfloat16

    if model_key == "flux2_klein":
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(cfg["model_id"], torch_dtype=dtype)
        pipe.enable_model_cpu_offload()

    elif model_key == "qwen_image":
        from diffusers import DiffusionPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=dtype
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {model_key}")

    return pipe


def generate_image(pipe, model_key: str, prompt: str, seed: int) -> Image.Image:
    """
    Run one inference call at the model's native resolution and return PIL Image
    downsampled to EVAL_SIZE × EVAL_SIZE.
    """
    cfg = MODEL_CONFIGS[model_key]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    h, w = cfg["gen_height"], cfg["gen_width"]

    if model_key == "flux2_klein":
        out = pipe(
            prompt=prompt,
            height=h,
            width=w,
            num_inference_steps=cfg["default_steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=generator,
        )

    elif model_key == "qwen_image":
        out = pipe(
            prompt=prompt,
            negative_prompt=cfg["negative_prompt"],
            height=h,
            width=w,
            num_inference_steps=cfg["default_steps"],
            true_cfg_scale=cfg["true_cfg_scale"],
            generator=generator,
        )

    img = out.images[0]

    # Downsample to eval resolution if needed
    if img.size != (EVAL_SIZE, EVAL_SIZE):
        img = img.resize((EVAL_SIZE, EVAL_SIZE), Image.LANCZOS)

    return img


# ---------------------------------------------------------------------------
# Main generation loop for one condition
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    model_key: str,
    pipe,
    prompts_dir: Path,
    out_dir: Path,
    seed: int,
) -> None:
    prompt_file = prompts_dir / condition / f"synthesis_{condition}.json"
    if not prompt_file.exists():
        print(f"  Prompt file not found: {prompt_file} — skipping")
        return

    with open(prompt_file, encoding="utf-8") as f:
        records = json.load(f)

    cond_out = out_dir / condition
    cond_out.mkdir(parents=True, exist_ok=True)

    skipped = generated = errors = 0

    for rec in tqdm(records, desc=f"{condition}"):
        fname  = safe_filename(rec["image_id"])
        out_path = cond_out / (Path(fname).stem + ".jpg")

        if out_path.exists():
            skipped += 1
            continue

        try:
            img = generate_image(pipe, model_key, rec["synthesized_caption"], seed)
            img.save(out_path, "JPEG", quality=95)
            generated += 1
        except Exception as e:
            print(f"  ERROR [{rec['image_id']}]: {e}")
            errors += 1

    print(f"  [{condition}] done — generated={generated}  skipped={skipped}  errors={errors}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: regenerate images from synthesized perspective prompts"
    )
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
        help="flux2_klein or qwen_image"
    )
    parser.add_argument(
        "--condition", default="all",
        help=f"Condition to run ({', '.join(ALL_CONDITIONS)}) or 'all' (default: all)"
    )
    parser.add_argument("--seed",        type=int, default=42,
                        help="Fixed seed for reproducibility (default: 42)")
    parser.add_argument("--prompts_dir", type=str, default=None,
                        help="Override synthesis prompts directory")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir \
                  else repo_root / "output" / "synthesis_prompts"
    out_dir     = Path(args.output_dir) if args.output_dir \
                  else repo_root / "output" / "regenerated_images" / args.model

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.condition == "all":
        conditions = ALL_CONDITIONS
    elif args.condition in ALL_CONDITIONS:
        conditions = [args.condition]
    else:
        parser.error(f"Unknown condition '{args.condition}'. Options: {ALL_CONDITIONS}")

    cfg = MODEL_CONFIGS[args.model]
    print(f"Model      : {args.model}  ({cfg['model_id']})")
    print(f"Generate at: {cfg['gen_width']}×{cfg['gen_height']}  →  saved at {EVAL_SIZE}×{EVAL_SIZE}")
    print(f"Steps      : {cfg['default_steps']}")
    print(f"Conditions : {conditions}")
    print(f"Output     : {out_dir}\n")

    print(f"Loading pipeline: {cfg['model_id']} ...")
    pipe = load_pipeline(args.model)
    print("Pipeline ready.\n")

    for cond in conditions:
        print(f"\n=== Condition: {cond} ===")
        run_condition(
            condition=cond,
            model_key=args.model,
            pipe=pipe,
            prompts_dir=prompts_dir,
            out_dir=out_dir,
            seed=args.seed,
        )

    print(f"\nPhase 2 complete. Images saved to: {out_dir}")


if __name__ == "__main__":
    main()
