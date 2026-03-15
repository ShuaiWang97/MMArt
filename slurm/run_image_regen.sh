#!/bin/bash
#
# Submit phase2_image_regen.py as SLURM jobs.
# Runs one condition per job for maximum parallelism.
#
# Usage:
#   bash slurm/run_image_regen.sh flux2_klein          # all 9 conditions
#   bash slurm/run_image_regen.sh qwen_image           # all 9 conditions
#   bash slurm/run_image_regen.sh flux2_klein NFE      # single condition
#
# Time estimates per condition (1000 images on H100):
#   flux2_klein : 4 steps  × ~1s/img  = ~17 min
#   qwen_image  : 25 steps × ~5s/img  = ~83 min
#

MODEL=${1:-flux2_klein}
CONDITION=${2:-all}

if [ "$CONDITION" == "all" ]; then
    CONDITIONS="N F E H NFE NFH NEH FEH NFEH"
else
    CONDITIONS="$CONDITION"
fi

# Walltime: flux2_klein is fast (1h buffer); qwen_image needs more (3h buffer)
if [ "$MODEL" == "flux2_klein" ]; then
    WALLTIME="01:00:00"
else
    WALLTIME="03:00:00"
fi

for COND in $CONDITIONS; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -n 10
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=${WALLTIME}
#SBATCH --job-name=regen_${MODEL}_${COND}
#SBATCH --output=/gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/slurm/slurm_regen_${MODEL}_${COND}_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd /projects/prjs0996/ArtRAG_Series/PolyArt
source activate gallery_gpt

python scripts/phase2_image_regen.py \
    --model ${MODEL} \
    --condition ${COND} \
    --seed 42
EOF
    echo "Submitted regen job: model=${MODEL}  condition=${COND}  walltime=${WALLTIME}"
done
