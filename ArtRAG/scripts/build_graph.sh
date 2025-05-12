#!/bin/bash

#SBATCH -n 10
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=60G
#SBATCH --job-name=hf_build_graph
#SBATCH --output=slurm/slurm_output_build_graph_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# Your job starts in the directory where you call sbatch 
cd ../../projects/0/prjs0996/PolyArt/ArtRAG
source activate LightRAG


export OPENAI_API_KEY=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
# python scripts/build_graph.py --working_dir Artpedia_gpt_4o_mini_ptompt_tuning --directory "../data/wikipedia_art/Artpedia_wiki_pages_test"
# python scripts/build_graph.py --working_dir SemArtv2_gpt_4o_ptompt_tuning --directory "../data/wikipedia_art/SemArtv2_wiki_pages_test"

# Start Ollama server in the background
# ollama serve > ollama_serve.log 2>&1 &
# python lightrag_ollama_age_demo.py


python scripts/build_graph_hg2.py 
# python scripts/build_graph.py --working_dir "./built_graph/All_gpt_4o_mini_prompt_tuning"  --directory "../data/wikipedia_art/ExpArt_wiki_pages_test/Artists"