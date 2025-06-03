# for job_id in  0 1 2 3 4 5 6 7
for job_id in  5
# for job_id in  0 
# for retrieval_strategy in "local"
do
    echo "Hello, Welcome for submiting job $job_id."
    sbatch scripts/inference_eval_job.sh $job_id
done

