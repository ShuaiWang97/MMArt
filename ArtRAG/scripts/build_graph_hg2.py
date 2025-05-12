import os
import sys
sys.path.append(os.path.abspath('.'))

from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_model_complete, hf_embedding, gpt_4o_mini_complete, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
import glob
from tqdm import tqdm
import argparse

WORKING_DIR = "./art_context"


def main(WORKING_DIR, model_name, directory):
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)

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

    # Check if the directory is a folder or a .txt file
    if os.path.isdir(directory):
        # List all .txt files in the directory
        txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

        # Iterate over each .txt file and process it
        for txt_file in tqdm(txt_files, desc="Processing files", unit="file"):
            try:
                with open(txt_file, 'r', encoding='utf8') as f:
                    content = f.read()
                    print(f"Processing file: {txt_file}")
                    rag.insert(content)
            except Exception as e:
                print(f"Error processing file {txt_file}: {str(e)}")
                continue
                
    elif os.path.isfile(directory) and directory.endswith('.txt'):
        # Process the single .txt file
        try:
            with open(directory, 'r', encoding='utf8') as f:
                content = f.read()
                print(f"Processing file: {directory}")
                rag.insert(content)
        except Exception as e:
            print(f"Error processing file {directory}: {str(e)}")
            return
    else:
        print(f"Error: {directory} is neither a directory nor a .txt file.")
        return

    print("------------------------------------------------------------------------------------")
    # prompt = {"text": "Can you help me generate the description of painting: \
    #             Maest (Cimabue), year: 1280", "image": "../data/Artpedia/Images/193.jpg"}
    # prompt = {"text": "Can you help me generate the description of painting: \
    #         Maest (Cimabue), year: 1280"}
    prompt = "Can you help me generate the description of painting: \
            Maest (Cimabue), year: 1280"
    # Perform naive search
    print(rag.query(prompt, param=QueryParam(mode="naive"), data_type="SemArtv2"))

    print("------------------------------------------------------------------------------------")
    # Perform hybrid search
    print(rag.query(prompt, param=QueryParam(mode="no-rag"), data_type="SemArtv2"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph with LightRAG.")
    parser.add_argument('--working_dir', type=str, 
                        default="./art_context", 
                        help='Working directory for LightRAG.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        choices=["Qwen/Qwen2.5-7B-Instruct"], help='LLM model function to use.')
    parser.add_argument('--directory', type=str, 
                        default="../../data/wikipedia_art/ExpArt_wiki_pages_test", 
                        help='Path to the directory containing .txt files for building graph.')

    args = parser.parse_args()


    main(args.working_dir, args.model_name, args.directory)