#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 6
#SBATCH -x bart14,bart13,eos20,eos21
#SBATCH --time=16:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out
python -m scripts.eval --wandb-proj coors-production --run-dir $SLURM_TMPDIR $@
cp -r  $SLURM_TMPDIR/wandb/* /network/tmp1/racaheva/coors/wandb


