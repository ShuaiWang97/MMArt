# PolyArt
Code base for PolyArt: A Multi-Perspective Dataset for Vision-Language Understanding in Visual Art


his repository contains code for generating descriptions of artwork using various Vision-Language Models (VLMs), including Qwen2-VL. The codebase supports processing artwork from datasets like WikiArt and generating detailed descriptions.

## Features
- Dataset sampling and balancing across art styles
- Image processing and description generation using Qwen2-VL




## Project Structure

├── data/
│ ├── wikiart/
│ │ ├── Images/
│ │ └── wikiart_full.csv
├──PolyArt
│ ├── scripts/
│ ├── ├── MLLM_inference.py
│ ├── ├── sample_data.py


### 1. Sample Balanced Dataset
To create a balanced sample of 200 paintings from the WikiArt dataset:

```bash
python scripts/sample_data.py
```

### 2. Generate Descriptions with Qwen2-VL
To generate descriptions for the sampled paintings:

```bash
python scripts/MLLM_inference.py \
    --csv_path "wikiart_balanced_200.csv" \
    --image_root_dir "../data/wikiart/Images" \
    --output_path "qwen2vl_captions.json"
```


## Model Support
- Qwen2-VL (7B-Instruct)
- Qwen2.5-VL (7B-Instruct)
- Other VLMs can be added by extending the model configuration


### Input CSV Format
The input CSV should contain the following columns:
- relative_path: Path to image file
- image: Image filename
- artist_attribution: Artist name
- style_classification: Art style
- title: Artwork title

### Output JSON Format
```json
[
    {
        "image_id": "path/to/image.jpg",
        "title": "Artwork Title",
        "artist": "Artist Name",
        "style": "Art Style",
        "generated_caption": "Generated description..."
    }
]
```