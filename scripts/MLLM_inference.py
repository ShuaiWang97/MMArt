import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json
from tqdm import tqdm
import os


def resize_if_needed(image, max_size=512):
    width, height = image.size
    if width > max_size or height > max_size:
        # Keep aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size/width))
        else:
            new_height = max_size
            new_width = int(width * (max_size/height))
        image = image.resize((new_width, new_height))
    return image


def process_paintings_with_qwen(csv_path, image_root_dir, output_path):
    # Load the model and processor
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load the CSV with sampled paintings
    df = pd.read_csv(csv_path)
    
    results = []
    
    # Process each painting
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load image
            image_path = os.path.join(image_root_dir, row['relative_path'])
            image = Image.open(image_path).convert('RGB')  # Added convert to RGB
            image = resize_if_needed(image)
            # Prepare prompt
            prompt = "Generate a concise description of this painting. Focus on essential elements such as *content* and *form*."
            
            # Prepare messages format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Process inputs correctly
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = inputs.to("cuda")

            # Generate caption
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_text = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Extract the actual response
            caption = generated_text[0].split("assistant")[-1].strip()
            print("generated explanations: ", caption)
            
            # Store results
            result = {
                "image_id": row['relative_path'],
                "title": row.get('image', ''),
                "artist": row.get('artist_attribution', ''),
                "style": row.get('style_classification', ''),
                "generated_caption": caption
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {row['relative_path']}: {str(e)}")
            continue
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    csv_path = "wikiart_balanced_200.csv"  # Path to your sampled CSV
    image_root_dir = "../data/wikiart/Images"  # Update with your image directory
    output_path = "output/qwen2-5vl_captions.json"
    
    process_paintings_with_qwen(csv_path, image_root_dir, output_path)