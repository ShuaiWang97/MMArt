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
        # llm_model_name='meta-llama/Llama-3.1-8B-Instruct',
        # llm_model_name="Qwen/Qwen2.5-72B-Instruct",
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

    # Load WikiArt dataset
    directory = "../wikiart_balanced_200.csv"
    print("Using WikiArt dataset")
    data = pd.read_csv(directory)[:args.data_num]

    # Prepare batch processing
    batch_size = 8  # Adjust based on your GPU memory
    results = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data.iloc[batch_start:batch_end]
        
        # Prepare batch messages
        batch_messages = []
        batch_image_paths = []
        
        for _, row in batch_data.iterrows():
            # Extract features for each painting
            tags = row['tags']
            artist = row['artist_name'] 
            img_id = row['image']
            style = row['style_classification']
            timeframe = row['timeframe_estimation']
            img_path = os.path.join("../../data/wikiart/Images", to_filesystem_name(row['relative_path']))
            
            # Create input for LightRAG model
            if args.question_type == "description":
                input_text = f"Please generate a description of this painting"
            elif args.question_type == "cultural&histroical":
                input_text = f"Please provide a historical and contextual analysis of the painting."

            input_text += f" with painting Metadata: painting name: {img_id}, Style: {style}, Artist: {artist}, Timeframe: {timeframe}, Tags: {tags}"
            
            # Add to batch
            batch_messages.append({"text": input_text, "image": img_path})

        # Run batch inference
        with torch.no_grad():
            generated_descriptions, retrieved_contexts, rerank_contexts = rag.query(
                batch_messages,
                param=QueryParam(mode=args.retrieval_strategy),
                data_type="WikiArt",
                shot_number=0
            )

        # Process results
        for i, (_, row) in enumerate(batch_data.iterrows()):
            results.append({
                'Image': row['image'],
                'Artist': row['artist_name'],
                'Style': row['style_classification'],
                'Timeframe': row['timeframe_estimation'],
                'Tags': row['tags'],
                'Generated Description': generated_descriptions[i],
                'Retrieved context': retrieved_contexts[i],
                'rerank_context': rerank_contexts[i]
            })
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()

        # Print GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"\nGPU Memory Usage:")
            print(f"Allocated: {gpu_memory_used:.2f} MB")
            print(f"Cached: {gpu_memory_cached:.2f} MB")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    output_DIR = os.path.join(WORKING_DIR, "output_{}_{}_{}data".format(
        current_date, "WikiArt", args.data_num))
    if not os.path.exists(output_DIR):
        os.mkdir(output_DIR)

    # Save the results and args to JSON files
    output_file = os.path.join(output_DIR,
                              'generated_descriptions_{}_{}.json'.format(args.retrieval_strategy, timestamp))
    results_df.to_json(output_file, orient='records',
                      indent=4, force_ascii=False)
    
    # Save args as JSON
    args_dict = vars(args)
    args_file = os.path.join(output_DIR,
                            'args_{}.json'.format(timestamp))
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
        default=200,
        help='Number of data samples to process'
    )

    parser.add_argument(
        '--question_type',
        type=str,
        default="cultural&histroical",
        choices=["description", "cultural&histroical", "Theme", "style&technique","Movement&school", "artist"],
        help='Type of question to generate'
    )

    args = parser.parse_args()
    print("args: ", args)



    generated_descriptions_file = run_ArtRAG_inference(args.working_dir, args.model_name, args)
