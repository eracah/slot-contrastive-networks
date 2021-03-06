#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH -x bart14,bart13,eos20,eos21
#SBATCH --time=16:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out
#SBATCH --partition=main                      # Ask for unkillable job

bash ./launch_scripts/mila_cluster.sl $@
