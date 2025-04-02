#!/bin/bash

#SBATCH -n 10
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=5:30:00
#SBATCH --mem=60G
#SBATCH --job-name=inference_art
#SBATCH --output=slurm/slurm_output_inference_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# Your job starts in the directory where you call sbatch 
cd ../../projects/0/prjs0996/PolyArt
source activate LightRAG

python scripts/MLLM_inference.py