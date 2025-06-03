#!/bin/bash

#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=inference_art
#SBATCH --output=slurm/slurm_output_offical_%A.out

module purge
# module load 2022
# module load Anaconda3/2022.05

module load 2023
module load Anaconda3/2023.07-2



# Your job starts in the directory where you call sbatch 
cd ../../projects/0/prjs0996/PolyArt/ArtRAG
source activate LightRAG

# module load Java/11.0.2

export OPENAI_API_KEY=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

export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPAT


python scripts/internVL_3.py

