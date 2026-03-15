#!/bin/bash
#
# Submit all 8 jobs for a given perspective:
#   bash slurm/run_perspectives.sh narrative
#   bash slurm/run_perspectives.sh formal
#   bash slurm/run_perspectives.sh emotional
#   bash slurm/run_perspectives.sh historical
#
# Each job covers ~9500 paintings; 8 jobs cover the full ~75k WikiArt dataset.

PERSPECTIVE=${1:?Usage: bash run_perspectives.sh [narrative|formal|emotional|historical]}

for JOB_ID in {0..7}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -n 10
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --job-name=polyart_${PERSPECTIVE}_${JOB_ID}
#SBATCH --output=/gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/slurm/slurm_${PERSPECTIVE}_%A_job${JOB_ID}.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd /gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt
source activate gallery_gpt

python scripts/generate_perspectives.py \
    --perspective ${PERSPECTIVE} \
    --job_id ${JOB_ID} \
    --csv_path /gpfs/work5/0/prjs0996/data/wikiart/wikiart_full.csv \
    --image_root_dir /gpfs/work5/0/prjs0996/data/wikiart/Images \
    --artemis_csv /gpfs/work5/0/prjs0996/ArtRAG_Series/PolyArt/output/artemis-v2/artemis_full.csv \
    --chunk_db_dir /projects/prjs0996/ArtRAG_Series/PolyArt/ArtRAG/art_context
EOF

    echo "Submitted ${PERSPECTIVE} job ${JOB_ID}"
done
