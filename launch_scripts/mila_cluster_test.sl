#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -x bart14,bart13,eos20,eos21
#SBATCH --time=24:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out
python -m scripts.train $@  #--run-dir $SLURM_TMPDIR
echo $WANDB_RUN_ID
#cp -r  $SLURM_TMPDIR/wandb/* /network/tmp1/racaheva/coors/wandb
#python -m scripts.eval --wandb-proj coors-scratch --run-dir $SLURM_TMPDIR/eval_runs --train-run-parent-dir $SLURM_TMPDIR/wandb --num-frames 2000 --epochs 2 --batch-size 32
#cp -r  $SLURM_TMPDIR/eval_runs/wandb/* /network/tmp1/racaheva/coors/wandb


