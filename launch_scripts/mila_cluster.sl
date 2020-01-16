#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATXH -x rtx6
#SBATCH --time=48:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out

## #SBATCH --partition=unkillable                      # Ask for unkillable job
# 1. Load your environment
# conda activate <env_name>

# 2. Copy your dataset on the compute node
#cp /network/data/<dataset> $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
echo $1
export module_name=$1
shift
echo $module_name
python -m $module_name  $@ --run-dir $SLURM_TMPDIR --final-dir /network/tmp1/racaheva/coors/wandb

# 4. Copy whatever you want to save on $SCRATCH
cp -r  $SLURM_TMPDIR/wandb/* /network/tmp1/racaheva/coors/wandb
