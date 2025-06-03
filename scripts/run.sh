
# for job_id in  0 1

for job_id in  2 3 4 5 6 7

do
    echo "Hello, Welcome for submiting job $job_id."
    sbatch scripts/run_job.sh $job_id
done

