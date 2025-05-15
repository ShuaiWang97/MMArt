#!/bin/bash

#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --time=50:00:00
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


# python scripts/inference_eval.py --model_name Qwen/Qwen2.5-VL-7B-Instruct
# python scripts/inference_eval.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --data_num 20

job_id=$1
python scripts/inference_eval.py --model_name Qwen/Qwen2.5-VL-72B-Instruct --job_id $job_id

# # Ask for question type as an argument
# question_type=$1
# python scripts/inference_eval.py --shot_number 0 --question_type $question_type --data_num 80


# python scripts/inference_eval.py --retrieval_strategy "local" --shot_number 2 --data_num 100 --data_type SemArtv1-context --working_dir "./built_graph/All_gpt_4o_mini_prompt_tuning"
# python scripts/inference_eval.py --data_type SemArtv2  --shot_number 2  --data_num 500  --working_dir "./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean"

# python scripts/inference_eval.py  --data_type SemArtv2  --generated_descriptions  ./built_graph/SemArtv2_gpt_4o_mini_prompt_tuning/output_2025-01-23_SemArtv2_100data/generated_descriptions_local_20250123_115023.json

# python scripts/vlm_inference.py --model_name "Qwen2_5-VL" --json_file_path "./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean/output_2025-03-11_SemArtv2_100data/generated_descriptions_20250311_134725.json"

