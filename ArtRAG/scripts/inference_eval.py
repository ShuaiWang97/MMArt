import clip
import torch
import ast
import logging
import pdb
from multiprocessing import Pool, cpu_count
import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
from pprint import pprint
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import json
import re
import argparse

from transformers import AutoModel, AutoTokenizer
from lightrag.utils import EmbeddingFunc
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete, hf_model_complete,hf_embedding
from lightrag import LightRAG, QueryParam, clip_score
import language_evaluation

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%m-%d")

def to_filesystem_name(s):
    # Replace each non-ASCII character with #Uxxxx
    return ''.join(
        c if ord(c) < 128 else f'#U{ord(c):04x}'
        for c in s
    )

def clean_description(description):
    """
    post processing on the generated text by deleting unrevelant symbols
    """
    # Remove markdown headers and formatting
    description = re.sub(r'###\s*Description of.*\n', '', description)
    description = re.sub(r'\*\*Content\*\*:', '', description)
    description = re.sub(r'\*\*Context\*\*:', '', description)
    description = re.sub(r'\*\*Form\*\*:', '', description)

    # Remove newline characters
    description = re.sub(r'\n', ' ', description)

    # Remove double quotes
    description = re.sub(r'"', '', description)

    # Remove other unwanted characters (e.g., **)
    description = re.sub(r'\*\*', '', description)

    # Normalize whitespace
    description = re.sub(r'\s+', ' ', description)

    # Remove leading and trailing whitespace
    description = description.strip()
    return description

def run_ArtRAG_inference(WORKING_DIR, model_name, args):
    """
    Run inference on the WikiArt dataset using the specified LightRAG model.
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,  
        llm_model_name=model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts, 
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            )
        ),
    )

    # Calculate data slice for this job
    total_data = 80000  # approximate total
    data_per_job = total_data // 8  # around 10,000 per job
    start_idx = args.job_id * data_per_job
    end_idx = start_idx + data_per_job if args.job_id < 7 else total_data
    
    # Load only the slice of data for this job
    data = pd.read_csv(args.data_source)[start_idx:end_idx]
    print(f"Processing data from index {start_idx} to {end_idx}")

    if args.data_source == "../data/wikiart/wikiart_full.csv":
        # Setup output directory and file
        output_DIR = os.path.join(WORKING_DIR, "output_all_{}_{}data".format(
            "WikiArt", args.data_num))
    elif args.data_source == "../wikiart_balanced_200.csv":
        # Setup output directory and file
        output_DIR = os.path.join(WORKING_DIR, "output_sample_200_{}_{}data".format(
            "WikiArt", args.data_num))
    if not os.path.exists(output_DIR):
        os.mkdir(output_DIR)


    output_file = os.path.join(output_DIR,
                            'generated_descriptions_{}_job_{}.json'.format(
                                args.retrieval_strategy, 
                                args.job_id))

    # Load existing results if available
    existing_results = []
    if os.path.exists(output_file):
        try:
            existing_results = pd.read_json(output_file).to_dict('records')
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Create a set of already processed image IDs
    processed_images = {result['Image'] for result in existing_results}
    
    # Placeholder for storing results
    results = existing_results.copy()
    
    # Iterate through each row in the dataset
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        img_id = row['image']
        
        # Skip if this image has already been processed
        if img_id in processed_images:
            print(f"Skipping already processed image: {img_id}")
            continue
            
        print("Processing row: ", index)
        
        # Extract required features for each painting
        tags = row['tags']
        artist = row['artist_name'] 
        style = row['style_classification']
        timeframe = row['timeframe_estimation']
        img = os.path.join("../../data/wikiart/Images", to_filesystem_name(row['relative_path']))

        # Create input for LightRAG model
        if args.question_type == "description":
            input_text = f"Please generate a description of this painting"
        elif args.question_type == "cultural&histroical":
            input_text = f"Please provide a historical and contextual analysis of the painting."

        input_text += f" with painting Metadata: painting name: {img_id}, Style: {style}, Artist: {artist}, Timeframe: {timeframe}, Tags: {tags}"
        query = {"text": input_text, "image": img}

        try:
            # Run inference with LightRAG model
            with torch.no_grad():
                generated_description, retrieved_context, rerank_context = rag.query(
                    query, param=QueryParam(mode=args.retrieval_strategy), data_type="WikiArt", shot_number=0)

            print("generated_description: ", generated_description)
            results.append({
                'Image': img_id,
                'Artist': artist,
                'Style': style,
                'Timeframe': timeframe,
                'Tags': tags,
                'Generated Description': generated_description,
                'Retrieved context': retrieved_context,
            })
            
            # Save progress every 10 images
            if len(results) % 100 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_json(output_file, orient='records',
                                indent=4, force_ascii=False)
                print(f"Progress saved: {len(results)} images processed")
            
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            # Save progress even if there's an error
            results_df = pd.DataFrame(results)
            results_df.to_json(output_file, orient='records',
                            indent=4, force_ascii=False)
            continue
        
        torch.cuda.empty_cache()

        # Print GPU memory usage
        if torch.cuda.is_available():
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
    results_df.to_json(output_file, orient='records',
                    indent=4, force_ascii=False)
    
    # Save args as JSON with job_id in filename
    args_dict = vars(args)
    args_file = os.path.join(output_DIR,
                            'args_{}_job_{}.json'.format(timestamp, args.job_id))
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Inference completed. Generated descriptions saved to '{output_file}'.")

    return output_file

def evaluate_batch(batch_predicts, batch_answers):
    try:
        evaluator = language_evaluation.CocoEvaluator()
        batch_result = evaluator.run_evaluation(batch_predicts, batch_answers)
        return batch_result
    except Exception as e:
        print(f"Error evaluating batch: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LightRAG inference and evaluate generated descriptions for WikiArt dataset.")
    
    # Model and data configuration
    parser.add_argument(
        '--working_dir', 
        type=str,
        default="./art_context",
        help='Working directory for LightRAG.'
    )
    parser.add_argument(
        '--model_name',
        type=str, 
        default='Qwen/Qwen2.5-VL-7B-Instruct',
        choices=['Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-32B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ', 'gpt_4o_complete'],
        help='LLM model function to use.'
    )
    
    # Inference settings
    parser.add_argument(
        '--retrieval_strategy',
        type=str,
        default="naive",
        choices=['local', 'global', 'hybrid', 'naive', 'no-rag'],
        help='Retrieval strategy to use.'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=10000,
        help='Number of data samples to process'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='../../data/wikiart/wikiart_full.csv',
        choices=['../wikiart_balanced_200.csv', '../../data/wikiart/wikiart_full.csv'],
        help='Source dataset to use for evaluation'
    )
    parser.add_argument(
        '--question_type',
        type=str,
        default="cultural&histroical",
        choices=["description", "cultural&histroical", "Theme", "style&technique","Movement&school", "artist"],
        help='Type of question to generate'
    )
    parser.add_argument(
        '--job_id',
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        help='Job ID (0-7) for parallel processing'
    )

    args = parser.parse_args()
    print("args: ", args)

    generated_descriptions_file = run_ArtRAG_inference(args.working_dir, args.model_name, args)
