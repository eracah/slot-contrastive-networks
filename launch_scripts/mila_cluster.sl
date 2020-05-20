#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -x bart14,bart13,eos20,eos21
#SBATCH --time=24:00:00
#SBATCH -o /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out  # Write the log on tmp1
#SBATCH -e /network/tmp1/racaheva/coors/slurm_stdout/slurm-%j.out

export WANDB_RUN_DIR=$SLURM_TMPDIR/wandb/train_run
export WANDB_TR_RUN_DIR=$WANDB_RUN_DIR
python -m scripts.train $@ --run-dir $WANDB_RUN_DIR
cp -r  $SLURM_TMPDIR/wandb/* /network/tmp1/racaheva/coors/wandb
export WANDB_RUN_DIR=$SLURM_TMPDIR/wandb/eval_run
python -m scripts.eval --wandb-proj coors-production  --id `cat ./wandb_id.txt` --tr-dir $WANDB_TR_RUN_DIR


