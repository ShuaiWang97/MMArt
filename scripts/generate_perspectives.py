"""
generate_perspectives.py

Generates e_narrative, e_formal, e_emotional, and e_historical perspectives for WikiArt paintings.

  e_narrative  <- Qwen3-VL-8B-Instruct (image + metadata, narrative prompt)
  e_formal     <- GalleryGPT  (LLaVA-7B + LoRA, formal analysis prompt)
  e_emotional  <- Qwen3-VL-8B-Instruct (image + metadata + ARTEMIS-v2 human utterances)
  e_historical <- Qwen3-VL-8B-Instruct (image + metadata + ArtRAG KG chunks via OpenAI embedding retrieval)

Usage:
  python scripts/generate_perspectives.py --perspective narrative   --job_id 0
  python scripts/generate_perspectives.py --perspective formal      --job_id 0
  python scripts/generate_perspectives.py --perspective emotional   --job_id 0
  python scripts/generate_perspectives.py --perspective historical  --job_id 0

Outputs:
  output/e_narrative/narrative_job_<job_id>.json
  output/e_formal/formal_job_<job_id>.json
  output/e_emotional/emotional_job_<job_id>.json
  output/e_historical/historical_job_<job_id>.json

Output schema per record:
  {
    "image_id":           str,   # relative_path (ASCII-safe)
    "title":              str,
    "artist":             str,
    "style":              str,
    "date":               str,
    "perspective":        str,   # "e_narrative" | "e_formal" | "e_emotional" | "e_historical"
    "generated_caption":  str,
    "dominant_emotion":   str,   # e_emotional only — majority vote from ARTEMIS-v2
    "artemis_coverage":   bool,  # e_emotional only — True if ARTEMIS utterances were available
    "artemis_utterances": list,  # e_emotional only — raw ARTEMIS utterances used as grounding
    "rag_query":          str,   # e_historical only — query string used for embedding retrieval
    "rag_chunks":         list,  # e_historical only — top-k retrieved KG chunk texts
  }
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. This script requires a CUDA-capable GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_filesystem_name(s: str) -> str:
    """Encode non-ASCII chars so the string is safe as a filesystem key."""
    return "".join(c if ord(c) < 128 else f"#U{ord(c):04x}" for c in s)


def load_existing(output_path: str) -> tuple[list, set]:
    results = []
    if os.path.exists(output_path):
        try:
            results = pd.read_json(output_path).to_dict("records")
            print(f"Resumed: {len(results)} existing records loaded from {output_path}")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh.")
    processed = {r["image_id"] for r in results}
    return results, processed


def save(results: list, output_path: str) -> None:
    pd.DataFrame(results).to_json(output_path, orient="records", indent=2)


def get_batch(df: pd.DataFrame, job_id: int, batch_size: int = 10_000) -> pd.DataFrame:
    start = job_id * batch_size
    end = min((job_id + 1) * batch_size, len(df))
    return df.iloc[start:end]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NARRATIVE_PROMPT = (
    'You are an art interpretation assistant.\n\n'
    'Given the painting titled "{title}" by {artist}, write a detailed '
    '**narrative and scene interpretation** — what is happening or might be '
    'happening in the scene. Focus on storytelling, implied action, '
    'relationships between figures, and atmosphere.\n\n'
    'Guidelines:\n'
    '- Length: ~80 words\n'
    '- Tone: descriptive and interpretive, not technical\n'
    '- Avoid: artistic terms (e.g. "chiaroscuro", "composition"), '
    'historical facts, or the artist\'s name\n\n'
    'Write the narrative and scene interpretation:'
)

FORMAL_PROMPT = (
    "Compose a short paragraph of formal analysis for this painting. "
    "Describe the composition, use of color and light, brushwork or technique, "
    "spatial organisation, and any notable visual effects. "
    "Focus purely on how the painting is made, not what it depicts or its historical context. "
    "Length: ~80 words."
)

EMOTIONAL_PROMPT_WITH_ARTEMIS = (
    'You are an art interpretation assistant.\n\n'
    'Look at the painting "{title}" by {artist}.\n\n'
    'Real viewers responded to this painting with the following emotional reactions:\n'
    '{utterances}\n\n'
    'The most common emotional response was: {dominant_emotion}.\n\n'
    'Using both what you see in the painting and these viewer reactions as grounding, '
    'write a coherent ~80-word **emotional interpretation** — the mood it evokes, '
    'the atmosphere, and the psychological tone. '
    'Synthesize the visual qualities of the painting with the viewer reactions into a unified emotional description. '
    'Write in third person (e.g. "The painting evokes..."), not first person.\n\n'
    'Write the emotional interpretation:'
)

EMOTIONAL_PROMPT_VISION_ONLY = (
    'You are an art interpretation assistant.\n\n'
    'Given the painting titled "{title}" by {artist}, write an ~80-word **emotional interpretation** '
    '— the mood it evokes, the atmosphere, and the psychological tone it creates in a viewer. '
    'Focus on emotional and affective qualities only. '
    'Avoid describing what is depicted or analysing technique. '
    'Write in third person (e.g. "The painting evokes...").\n\n'
    'Write the emotional interpretation:'
)


# ---------------------------------------------------------------------------
# Narrative pipeline  (Qwen2.5-VL / Qwen3-VL)
# ---------------------------------------------------------------------------

def run_narrative(args, df_batch, results, processed, output_path):
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading {args.model_name} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"narrative job {args.job_id}"):
        image_id = to_filesystem_name(row["relative_path"])
        if image_id in processed:
            continue

        image_path = os.path.join(args.image_root_dir, row["relative_path"])
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            prompt_text = NARRATIVE_PROMPT.format(
                title=row.get("image", "Untitled"),
                artist=row.get("artist_name", "Unknown Artist"),
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].split("assistant")[-1].strip()

            results.append({
                "image_id":          image_id,
                "title":             row.get("image", ""),
                "artist":            row.get("artist_name", ""),
                "style":             row.get("style_classification", ""),
                "date":              str(row.get("date", "")),
                "perspective":       "e_narrative",
                "generated_caption": caption,
            })
            processed.add(image_id)

            if len(results) % 100 == 0:
                save(results, output_path)
                print(f"  checkpoint: {len(results)} records")

        except Exception as e:
            print(f"Error on {row['relative_path']}: {e}")
            save(results, output_path)

        torch.cuda.empty_cache()

    save(results, output_path)


# ---------------------------------------------------------------------------
# Formal pipeline  (GalleryGPT: LLaVA-7B + LoRA)
# ---------------------------------------------------------------------------

def run_formal(args, df_batch, results, processed, output_path):
    # GalleryGPT uses a custom llava package — add it to path
    gallery_gpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GalleryGPT")
    sys.path.insert(0, gallery_gpt_dir)

    from llava.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    print(f"Loading GalleryGPT from {args.model_path} (base: {args.model_base}) ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Conversation mode for ShareGPT4V / LLaVA v1 family
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # Build query string with image token
    qs = FORMAL_PROMPT
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"formal job {args.job_id}"):
        image_id = to_filesystem_name(row["relative_path"])
        if image_id in processed:
            continue

        image_path = Path(args.image_root_dir) / row["relative_path"]
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                    max_new_tokens=256,
                    use_cache=True,
                )

            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            results.append({
                "image_id":          image_id,
                "title":             row.get("image", ""),
                "artist":            row.get("artist_name", ""),
                "style":             row.get("style_classification", ""),
                "date":              str(row.get("date", "")),
                "perspective":       "e_formal",
                "generated_caption": caption,
            })
            processed.add(image_id)

            if len(results) % 100 == 0:
                save(results, output_path)
                print(f"  checkpoint: {len(results)} records")

        except Exception as e:
            print(f"Error on {row['relative_path']}: {e}")
            save(results, output_path)

        torch.cuda.empty_cache()

    save(results, output_path)


# ---------------------------------------------------------------------------
# Emotional pipeline  (Qwen2.5-VL-7B + ARTEMIS-v2 utterances as grounding)
# ---------------------------------------------------------------------------

def load_artemis_index(artemis_csv: str, max_utterances: int = 6) -> dict:
    """
    Returns a dict keyed by painting name (lowercase) with:
      {
        "dominant_emotion": str,
        "utterances": [str, ...]   # up to max_utterances, diverse sample
      }
    """
    print(f"Loading ARTEMIS-v2 from {artemis_csv} ...")
    df = pd.read_csv(artemis_csv)

    index = {}
    for painting, group in df.groupby("painting"):
        # Dominant emotion = majority vote
        dominant = group["emotion"].value_counts().index[0]

        # Sample utterances: prefer grounding_emotion matches dominant, then others
        grounded = group[group["grounding_emotion"] == dominant]["utterance_spelled"].tolist()
        others   = group[group["grounding_emotion"] != dominant]["utterance_spelled"].tolist()

        # Mix: up to 4 from dominant emotion, up to 2 from others for diversity
        selected = grounded[:4] + others[:2]
        # Deduplicate while preserving order
        seen, unique = set(), []
        for u in selected:
            if isinstance(u, str) and u.lower() not in seen:
                seen.add(u.lower())
                unique.append(u)

        index[painting.lower()] = {
            "dominant_emotion": dominant,
            "utterances": unique[:max_utterances],
        }

    print(f"  ARTEMIS index built: {len(index):,} paintings")
    return index


def painting_key(relative_path: str) -> str:
    """Extract painting name from relative_path (e.g. 'Style/artist_title.jpg' → 'artist_title')."""
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    return stem.lower()


def run_emotional(args, df_batch, results, processed, output_path):
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Build ARTEMIS lookup index
    artemis_index = load_artemis_index(args.artemis_csv)

    print(f"Loading {args.model_name} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    artemis_hits, artemis_misses = 0, 0

    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"emotional job {args.job_id}"):
        image_id = to_filesystem_name(row["relative_path"])
        if image_id in processed:
            continue

        image_path = os.path.join(args.image_root_dir, row["relative_path"])
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            title  = row.get("image", "Untitled")
            artist = row.get("artist_name", "Unknown Artist")

            # Look up ARTEMIS utterances
            key = painting_key(row["relative_path"])
            artemis_entry = artemis_index.get(key)
            has_artemis = artemis_entry is not None and len(artemis_entry["utterances"]) > 0

            if has_artemis:
                artemis_hits += 1
                utterance_lines = "\n".join(f'- "{u}"' for u in artemis_entry["utterances"])
                prompt_text = EMOTIONAL_PROMPT_WITH_ARTEMIS.format(
                    title=title,
                    artist=artist,
                    utterances=utterance_lines,
                    dominant_emotion=artemis_entry["dominant_emotion"],
                )
                dominant_emotion = artemis_entry["dominant_emotion"]
            else:
                artemis_misses += 1
                prompt_text = EMOTIONAL_PROMPT_VISION_ONLY.format(title=title, artist=artist)
                dominant_emotion = ""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].split("assistant")[-1].strip()

            results.append({
                "image_id":           image_id,
                "title":              title,
                "artist":             artist,
                "style":              row.get("style_classification", ""),
                "date":               str(row.get("date", "")),
                "perspective":        "e_emotional",
                "generated_caption":  caption,
                "dominant_emotion":   dominant_emotion,
                "artemis_coverage":   has_artemis,
                "artemis_utterances": artemis_entry["utterances"] if has_artemis else [],
            })
            processed.add(image_id)

            if len(results) % 100 == 0:
                save(results, output_path)
                print(f"  checkpoint: {len(results)} records "
                      f"(ARTEMIS: {artemis_hits} hits, {artemis_misses} misses)")

        except Exception as e:
            print(f"Error on {row['relative_path']}: {e}")
            save(results, output_path)

        torch.cuda.empty_cache()

    save(results, output_path)
    print(f"  Final — ARTEMIS grounded: {artemis_hits} | vision-only fallback: {artemis_misses}")


HISTORICAL_PROMPT = (
    'You are an art historian.\n\n'
    'The following context has been retrieved from an art knowledge base about "{title}" by {artist} ({style}, {date}):\n'
    '{context}\n\n'
    'Using the retrieved context and your knowledge of art history, '
    'write a coherent ~80-word **historical and cultural interpretation** of this painting — '
    'covering the artistic movement, historical period, cultural setting, and any relevant '
    'influences, patronage, or significance.\n\n'
    'Guidelines:\n'
    '- Focus on history and cultural context only\n'
    '- Do NOT describe visual elements, colours, brushwork, or composition\n'
    '- Do NOT speculate on specific details not supported by the context or established art history\n'
    '- Write in third person (e.g. "The painting reflects...")\n\n'
    'Write the historical interpretation:'
)

HISTORICAL_PROMPT_NO_RAG = (
    'You are an art historian.\n\n'
    'Write a ~80-word **historical and cultural interpretation** of "{title}" by {artist} ({style}, {date}) — '
    'covering the artistic movement, historical period, cultural setting, and any relevant '
    'influences or significance.\n\n'
    'Guidelines:\n'
    '- Focus on history and cultural context only\n'
    '- Do NOT describe visual elements, colours, brushwork, or composition\n'
    '- If you are uncertain about specific facts for this artist or work, speak at the level of the movement and period\n'
    '- Write in third person (e.g. "The painting reflects...")\n\n'
    'Write the historical interpretation:'
)


def load_chunk_index(chunk_db_dir: str) -> dict:
    """
    Load art-history text chunk embeddings from art_context vdb_chunks.json.

    Returns a dict with:
      matrix    : np.ndarray [N x 384]  — L2-normalised chunk embeddings (sentence-transformers/all-MiniLM-L6-v2)
      chunk_ids : list[str]             — chunk ID per row (matches kv_store_text_chunks.json keys)
      kv_store  : dict[str, dict]       — chunk_id → {"content": str, "tokens": int}
    """
    import base64
    import numpy as np

    print(f"Loading art-context chunk index from {chunk_db_dir} ...")

    vdb_path = os.path.join(chunk_db_dir, "vdb_chunks.json")
    vdb = json.load(open(vdb_path))
    matrix_bytes = base64.b64decode(vdb["matrix"])
    n   = len(vdb["data"])
    dim = vdb["embedding_dim"]  # 384
    matrix    = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(n, dim).copy()
    chunk_ids = [d["__id__"] for d in vdb["data"]]

    kv_path  = os.path.join(chunk_db_dir, "kv_store_text_chunks.json")
    kv_store = json.load(open(kv_path))

    print(f"  Chunks: {n} | Embedding dim: {dim} | KV entries: {len(kv_store)}")
    return {"matrix": matrix, "chunk_ids": chunk_ids, "kv_store": kv_store}


def retrieve_chunk_context(query: str, chunk_index: dict, embed_model,
                           top_k: int = 3, sim_threshold: float = 0.25) -> tuple[list, float]:
    """
    Naive chunk retrieval:
      1. Embed query with local sentence-transformer (384-dim, no API cost)
      2. Cosine similarity against all art-history chunks
      3. If best similarity < sim_threshold, return empty (no relevant chunk found)
      4. Return top-k chunk texts + max similarity score

    Returns: (context_texts, max_similarity)
    """
    import numpy as np

    matrix    = chunk_index["matrix"]
    chunk_ids = chunk_index["chunk_ids"]
    kv_store  = chunk_index["kv_store"]

    # Embed query with local model (fast, ~2ms on CPU)
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims  = matrix @ q_vec   # cosine similarity (matrix rows are already normalized)

    max_sim = float(sims.max())
    if max_sim < sim_threshold:
        return [], max_sim

    top_idx = sims.argsort()[::-1][:top_k]
    chunks  = []
    for i in top_idx:
        if sims[i] < sim_threshold:
            break
        cid     = chunk_ids[i]
        content = kv_store.get(cid, {}).get("content", "").strip()
        if content:
            chunks.append(content[:800])  # truncate very long chunks

    return chunks, max_sim


def run_historical(args, df_batch, results, processed, output_path):
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from sentence_transformers import SentenceTransformer

    # Load art-history chunk index (CPU, runs once)
    chunk_index = load_chunk_index(args.chunk_db_dir)

    # Load local embedding model (384-dim, no API cost)
    print("Loading sentence-transformer embedding model ...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    print(f"Loading {args.model_name} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    rag_hits, rag_misses = 0, 0

    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch),
                         desc=f"historical job {args.job_id}"):
        image_id = row["relative_path"]
        if image_id in processed:
            continue
        try:
            image_path = os.path.join(args.image_root_dir, image_id)
            if not os.path.exists(image_path):
                print(f"Missing image: {image_path}")
                continue

            image  = Image.open(image_path).convert("RGB")
            title  = row.get("image", "Untitled")
            artist = row.get("artist_name", "Unknown Artist")
            style  = row.get("style_classification", "")
            date   = str(row.get("date", ""))

            # Build natural language query from all available metadata
            def _val(key):
                v = row.get(key)
                return str(v).replace("_", " ") if v and str(v) != "nan" else ""

            # Extract painting name (strip artist prefix: "artist-name_painting-name.jpg" → "painting name")
            stem = re.sub(r"\.(jpg|jpeg|png)$", "", title, flags=re.IGNORECASE)
            parts = stem.split("_", 1)
            painting_name = parts[1].replace("-", " ").strip() if len(parts) > 1 else stem.replace("-", " ").strip()

            extra_styles = _val("additional_styles")
            school_val   = _val("artist_school")
            tags_str     = _val("tags").replace(",", ", ") if _val("tags") else ""

            # Natural language query for chunk retrieval
            # The art_context chunks are Wikipedia art-history paragraphs, so
            # a descriptive query about the style/movement/period retrieves well
            style_clean = style.replace("_", " ")
            query_parts = [f"{style_clean} painting"]
            if extra_styles:
                query_parts.append(extra_styles)
            if school_val:
                query_parts.append(f"{school_val} art")
            if date and date != "nan":
                query_parts.append(date[:4])   # just the year
            query_parts.append("historical cultural context art movement")
            rag_query = " ".join(query_parts)

            rag_chunks, rag_sim = retrieve_chunk_context(
                rag_query, chunk_index, embed_model, top_k=args.rag_top_k
            )

            if rag_chunks:
                rag_hits += 1
                context_text = "\n\n".join(
                    f"[Context {i+1}]: {c}" for i, c in enumerate(rag_chunks)
                )
                prompt_text = HISTORICAL_PROMPT.format(
                    title=title, artist=artist, style=style,
                    date=date, context=context_text,
                )
            else:
                rag_misses += 1
                prompt_text = HISTORICAL_PROMPT_NO_RAG.format(
                    title=title, artist=artist, style=style, date=date,
                )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].split("assistant")[-1].strip()

            results.append({
                "image_id":          image_id,
                "title":             title,
                "artist":            artist,
                "style":             style,
                "date":              date,
                "perspective":       "e_historical",
                "generated_caption": caption,
                "rag_query":         rag_query,
                "rag_sim":           round(rag_sim, 4),
                "rag_chunks":        rag_chunks,
            })
            processed.add(image_id)

            if len(results) % 100 == 0:
                save(results, output_path)
                hit_rate = rag_hits / (rag_hits + rag_misses) * 100 if (rag_hits + rag_misses) else 0
                print(f"  checkpoint: {len(results)} records "
                      f"(RAG hits: {rag_hits} [{hit_rate:.0f}%], vision-only: {rag_misses})")

        except Exception as e:
            print(f"Error on {row['relative_path']}: {e}")
            save(results, output_path)

        torch.cuda.empty_cache()

    save(results, output_path)
    print(f"  Final — RAG grounded: {rag_hits} | vision-only fallback: {rag_misses}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate e_narrative, e_formal, e_emotional, or e_historical perspectives for WikiArt")
    parser.add_argument("--perspective", required=True, choices=["narrative", "formal", "emotional", "historical"],
                        help="Which perspective to generate")
    parser.add_argument("--job_id", type=int, required=True, choices=range(8),
                        help="Job index (0-7); each job processes 10k paintings")
    parser.add_argument("--csv_path", type=str,
                        default="data/wikiart/wikiart_full.csv")
    parser.add_argument("--image_root_dir", type=str,
                        default="data/wikiart/Images")

    # Qwen (narrative + emotional)
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model name for narrative/emotional (Qwen3-VL recommended)")

    # GalleryGPT (formal)
    parser.add_argument("--model_path", type=str,
                        default="GalleryGPT/llava-lora-model",
                        help="Path to GalleryGPT LoRA checkpoint")
    parser.add_argument("--model_base", type=str,
                        default="GalleryGPT/share4v/llava-7b",
                        help="Path to base LLaVA-7B model for LoRA loading")

    # ARTEMIS-v2 (emotional)
    parser.add_argument("--artemis_csv", type=str,
                        default="output/artemis-v2/combined/train/artemis_preprocessed.csv",
                        help="Path to ARTEMIS-v2 preprocessed CSV")
    parser.add_argument("--artemis_max_utterances", type=int, default=6,
                        help="Max ARTEMIS utterances to include per painting (default: 6)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N paintings (for quick testing)")

    # Art-context chunk DB (historical) — naive chunk retrieval with local sentence-transformer
    parser.add_argument("--chunk_db_dir", type=str,
                        default="/projects/prjs0996/ArtRAG_Series/PolyArt/ArtRAG/art_context",
                        help="Path to art_context directory (contains vdb_chunks.json + kv_store_text_chunks.json)")
    parser.add_argument("--rag_top_k", type=int, default=3,
                        help="Number of art-history chunks to retrieve per painting (default: 3)")

    args = parser.parse_args()

    # Resolve paths relative to repo root (one level up from scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.csv_path        = os.path.join(repo_root, args.csv_path)
    args.image_root_dir  = os.path.join(repo_root, args.image_root_dir)
    if not os.path.isabs(args.model_path):
        args.model_path  = os.path.join(repo_root, args.model_path)
    if not os.path.isabs(args.model_base):
        args.model_base  = os.path.join(repo_root, args.model_base)
    if not os.path.isabs(args.artemis_csv):
        args.artemis_csv = os.path.join(repo_root, args.artemis_csv)

    # Output path
    perspective_key = f"e_{args.perspective}"
    output_dir = os.path.join(repo_root, "output", perspective_key)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.perspective}_job_{args.job_id}.json")

    # Load data
    df = pd.read_csv(args.csv_path)
    df_batch = get_batch(df, args.job_id)
    if args.limit:
        df_batch = df_batch.head(args.limit)
    print(f"Perspective: {perspective_key} | Job {args.job_id} | "
          f"Rows {args.job_id * 10_000}–{args.job_id * 10_000 + len(df_batch) - 1} "
          f"({len(df_batch)} paintings)")

    results, processed = load_existing(output_path)

    # Save run metadata
    meta = {
        "perspective": perspective_key,
        "job_id": args.job_id,
        "model": args.model_path if args.perspective == "formal" else args.model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(output_dir, f"args_job_{args.job_id}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.perspective == "narrative":
        run_narrative(args, df_batch, results, processed, output_path)
    elif args.perspective == "formal":
        run_formal(args, df_batch, results, processed, output_path)
    elif args.perspective == "emotional":
        run_emotional(args, df_batch, results, processed, output_path)
    else:
        run_historical(args, df_batch, results, processed, output_path)

    print(f"Done. {len(results)} total records → {output_path}")


if __name__ == "__main__":
    main()
