import time
from collections import deque
from itertools import chain

import numpy as np
import torch
from src.utils import get_argparser
import wandb
import os
from atariari.benchmark.episodes import get_episodes


def train_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if str(device) != "cuda" and os.environ["HOME"] != '/Users/evanracah':
        assert False, "device must be cuda!"
    tr_eps, val_eps = get_episodes(steps=args.num_frames,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.probe_collect_mode,
                                 train_mode="train_encoder",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)

    observation_shape = tr_eps[0][0].shape
    torch.set_num_threads(1)
    args.obs_space = observation_shape
    config = {}
    config.update(vars(args))
    if args.method == "nce":
        from src.slot_nce import NCETrainer
        trainer = NCETrainer(args, device=device, wandb=wandb)

    else:
        assert False, "method {} has no trainer".format(args.method)

    encoder = trainer.train(tr_eps, val_eps)

    return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    train_encoder(args)
