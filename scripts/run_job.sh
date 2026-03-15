#!/bin/bash

#SBATCH -n 10
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --job-name=inference_art
#SBATCH --output=slurm/slurm_output_inference_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch 
cd ../../projects/0/prjs0996/PolyArt
source activate LightRAG


job_id=$1
python scripts/MLLM_inference.py --model_name Qwen/Qwen2.5-VL-72B-Instruct --job_id $job_id 

# python scripts/MLLM_inference.py --model_name Qwen/Qwen2.5-VL-72B-Instruct --csv_path wikiart_balanced_200.csv