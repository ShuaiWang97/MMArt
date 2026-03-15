#!/bin/bash
#SBATCH --job-name=polyart_unified
#SBATCH --output=/gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/slurm/slurm_unified_%j.out
#SBATCH --time=16:00:00
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -p gpu_h100

module purge
module load 2023
module load Anaconda3/2023.07-2

cd /gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt
source activate gallery_gpt

python scripts/phase4_unified.py --batch_size 64
