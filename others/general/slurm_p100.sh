#!/bin/bash
#SBATCH --mail-user=<x5bai@uwaterloo.ca>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="p100_job"
#SBATCH --partition=gpu_p100
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

srun python /work/x5bai/project/Code_Files/general/op_runner_singleproc.py hydra.run.dir=/work/x5bai/project/Data_Files/ldc_2d_$(date +%s)
