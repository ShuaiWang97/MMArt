import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json
from tqdm import tqdm
import os
import argparse
from datetime import datetime


# def resize_if_needed(image, max_size=512):
#     width, height = image.size
#     if width > max_size or height > max_size:
#         # Keep aspect ratio
#         if width > height:
#             new_width = max_size
#             new_height = int(height * (max_size/width))
#         else:
#             new_height = max_size
#             new_width = int(width * (max_size/height))
#         image = image.resize((new_width, new_height))
#     return image

def to_filesystem_name(s):
    # Replace each non-ASCII character with #Uxxxx
    return ''.join(
        c if ord(c) < 128 else f'#U{ord(c):04x}'
        for c in s
    )
    
def process_paintings_with_qwen(csv_path, image_root_dir, output_path, model_name, job_id):
    # Load the model and processor
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Load the CSV with sampled paintings
    df = pd.read_csv(csv_path)
    
    # Calculate batch indices
    batch_size = 10000
    start_idx = job_id * batch_size
    end_idx = min((job_id + 1) * batch_size, len(df))
    
    # Get the batch of paintings
    df_batch = df.iloc[start_idx:end_idx]
    
    # Load existing results if available
    existing_results = []
    if os.path.exists(output_path):
        try:
            existing_results = pd.read_json(output_path).to_dict('records')
            print(f"Loaded {len(existing_results)} existing results from {output_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Create a set of already processed image IDs
    processed_images = {result['image_id'] for result in existing_results}
    
    # Initialize results with existing ones
    results = existing_results.copy()
    
    # Process each painting
    for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Processing job {job_id}"):
        try:
            # Skip if already processed
            if row['relative_path'] in processed_images:
                print(f"Skipping already processed image: {row['relative_path']}")
                continue
                
            # Load image
            image_path = os.path.join(image_root_dir, row['relative_path'])
            image = Image.open(image_path).convert('RGB')
            
            print("image: ", row.get('image', 'Untitled'))
            print("artist: ", row.get('artist_attribution', 'Unknown Artist'))
            
            prompt = f"""
            You are an art interpretation assistant.

            Given a painting titled "{row.get('image', 'Untitled')}" by {row.get('artist_attribution', 'Unknown Artist')}, generate a detailed **narrative and scene interpretation** that explains what is happening or might be happening in the scene. Focus on storytelling, implied action, relationships, and atmosphere. Avoid analyzing the visual style or historical context—focus on the **implied story** within the image.

            --- Target response style and length ---
            - Length: around 80 words
            - Tone: Descriptive and interpretive, not technical
            - Avoid: Artistic terms (e.g., "chiaroscuro," "composition"), historical facts, or artist names

            Below is the input:

            [IMAGE]

            Now write the narrative and scene interpretation:
            """
            
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
            )
            inputs = inputs.to("cuda")

            # Generate caption
            generated_ids = model.generate(**inputs, max_new_tokens=512)
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
                "image_id": to_filesystem_name(row['relative_path']),
                "title": row.get('image', ''),
                "artist": row.get('artist_attribution', ''),
                "style": row.get('style_classification', ''),
                "generated_caption": caption
            }
            results.append(result)
            
            # Save progress every 100 images
            if len(results) % 100 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_json(output_path, orient='records', indent=2)
                print(f"Progress saved: {len(results)} images processed")
            
        except Exception as e:
            print(f"Error processing {row['relative_path']}: {str(e)}")
            # Save progress even if there's an error
            results_df = pd.DataFrame(results)
            results_df.to_json(output_path, orient='records', indent=2)
            continue
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Print GPU memory usage
        if torch.cuda.is_available() and idx==0:
            print(f"\nGPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
                print(f"GPU {i}:")
                print(f"  Allocated: {gpu_memory_used:.2f} MB")
                print(f"  Cached: {gpu_memory_cached:.2f} MB")
    
    # Final save of results
    results_df = pd.DataFrame(results)
    results_df.to_json(output_path, orient='records', indent=2)
    
    # Save args as JSON
    args_dict = {
        "model_name": model_name,
        "job_id": job_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    args_file = os.path.join(os.path.dirname(output_path), f'args_job_{job_id}.json')
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print(f"Processing completed. Results saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process paintings with Qwen VL model')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                      choices=["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"],
                      help='Name of the Qwen VL model to use')
    parser.add_argument('--job_id', type=int, 
                      default=0,
                      choices=[0, 1, 2, 3, 4, 5, 6, 7],
                      help='Job ID (0-7) for parallel processing')
    parser.add_argument('--csv_path', type=str, default="../data/wikiart/wikiart_full.csv",
                      help='Path to the CSV file containing painting metadata')
    args = parser.parse_args()
    
    # Create timestamp for output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.csv_path == "../data/wikiart/wikiart_full.csv":
        output_dir = f"output/narrative_interpretation"
    elif args.csv_path == "wikiart_balanced_200.csv":
        output_dir = f"output_sample_200/narrative_interpretation_balanced_200"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = args.csv_path
    image_root_dir = "../data/wikiart/Images"
    output_path = f"{output_dir}/qwen2-5vl_captions_job_{args.job_id}.json"
    
    process_paintings_with_qwen(csv_path, image_root_dir, output_path, args.model_name, args.job_id)