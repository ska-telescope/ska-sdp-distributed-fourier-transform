#!/usr/bin/env bash
# Script to submit multiple runs

for size in 1 10 20 50 100 200 300 500 800 1000 2000 3000 5000 8000 10000
do
    echo "Running queue size: $size"
    sbatch run_distr_single_csd3.slurm $size
done
