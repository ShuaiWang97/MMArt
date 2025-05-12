import sys
import os
sys.path.append(os.path.abspath('.'))

import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import glob
from tqdm import tqdm


def main(WORKING_DIR, llm_model_func, directory):
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func  # Use the specified LLM model function
    )

    # Check if the directory is a folder or a .txt file
    if os.path.isdir(directory):
        # List all .txt files in the directory
        txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

        # Iterate over each .txt file and process it
        for txt_file in tqdm(txt_files, desc="Processing files", unit="file"):
            with open(txt_file, 'r', encoding='utf8') as f:
                content = f.read()
                print(f"Processing file: {txt_file}")
                rag.insert(content)
    elif os.path.isfile(directory) and directory.endswith('.txt'):
        # Process the single .txt file
        with open(directory, 'r', encoding='utf8') as f:
            content = f.read()
            print(f"Processing file: {directory}")
            rag.insert(content)
    else:
        print(f"Error: {directory} is neither a directory nor a .txt file.")
        return
    print("------------------------------------------------------------------------------------")
    prompt = {"text": "Can you help me generate the description of painting: \
                Maest (Cimabue), year: 1280", "image": "../data/Artpedia/Images/193.jpg"}
    # Perform naive search
    print(rag.query(prompt, param=QueryParam(mode="naive"),data_type="SemArtv2"))

    print("------------------------------------------------------------------------------------")
    # Perform hybrid search
    print(rag.query(prompt, param=QueryParam(mode="no-rag"),data_type="SemArtv2"))

    print(rag.query(prompt, param=QueryParam(mode="local"),data_type="SemArtv1-context"))

    # print("------------------------------------------------------------------------------------")
    # Perform hybrid search with text
    print(rag.query(prompt["text"],  param=QueryParam(mode="local"),data_type="SemArtv2"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph with LightRAG.")
    parser.add_argument('--working_dir', type=str, 
                        default = "./built_graph/All_gpt_4o_mini_prompt_tuning", 
                        help='Working directory for LightRAG.')
    parser.add_argument('--llm_model_func', type=str,  default='gpt_4o_mini_complete',
                        choices=['gpt_4o_mini_complete', 'gpt_4o_complete'], help='LLM model function to use.')
    parser.add_argument('--directory', type=str, 
                        default="../data/wikipedia_art/Artpedia_wiki_pages_test", help='Path to the directory containing .txt files for building graph.')

    args = parser.parse_args()

    # Map the llm_model_func argument to the actual function
    llm_model_func_map = {
        'gpt_4o_mini_complete': gpt_4o_mini_complete,
        'gpt_4o_complete': gpt_4o_complete
    }

    main(args.working_dir, llm_model_func_map[args.llm_model_func], args.directory)