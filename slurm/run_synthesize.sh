#!/bin/bash
#
# Submit phase1_synthesize.py as a single SLURM job.
# Singles (N/F/E/H) are copied in-process; multi conditions run through vLLM.
#
# Usage:
#   bash slurm/run_synthesize.sh            # all 9 conditions
#   bash slurm/run_synthesize.sh NFE        # single condition
#

CONDITION=${1:-all}

sbatch <<EOF
#!/bin/bash
#SBATCH -n 10
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=polyart_synthesize_${CONDITION}
#SBATCH --output=/gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/slurm/slurm_synthesize_${CONDITION}_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd /projects/prjs0996/ArtRAG_Series/PolyArt
source activate gallery_gpt

python scripts/phase1_synthesize.py \
    --condition ${CONDITION} \
    --n_sample 1000 \
    --model_name Qwen/Qwen3-8B \
    --batch_size 512 \
    --gpu_mem 0.85
EOF

echo "Submitted synthesis job (condition=${CONDITION})"
