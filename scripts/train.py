from atariari.benchmark.episodes import get_episodes
from src.utils import  append_suffix, print_memory, get_channels
import shutil
import argparse
import os
import torch
from pathlib import Path
import wandb
# methods that need encoder trained before
ablations = ["nce", "hybrid", "loss1_only", "loss2_only", "none"]
baselines = ["supervised", "random-cnn", "stdim", "cswm"]


def get_argparser():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument("--final-dir", type=str, default=".")
    parser.add_argument('--num-frames', type=int, default=100000,  help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"], default="random_agent")
    parser.add_argument('--num-processes', type=int, default=8, help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4', help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=1, help='Number of frames to stack for a state')
    parser.add_argument('--no-downsample', action='store_true', default=True, help='Whether to use a linear classifier')
    parser.add_argument("--color", action='store_true', default=True)
    parser.add_argument("--checkpoint-index", type=int, default=-1)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument('--method', type=str, default='scn', choices= baselines + ["scn"], help='Method to use for training representations (default: scn')
    parser.add_argument('--ablation', type=str, default="none", choices=ablations, help='Ablation of scn (default: scn')
    parser.add_argument("--end-with-relu", action='store_true', default=False)
    parser.add_argument('--encoder-type', type=str, default="stdim", choices=["stdim", "cswm"], help='Encoder type stim or cswm')
    parser.add_argument('--feature-size', type=int, default=256, help='Size of features')
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--slot-len", type=int, default=64)
    parser.add_argument("--fmap-num", default="f7")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate foe learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for  (default: 100)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args, unknown = parser.parse_known_args()
    args.num_channels = get_channels(args)
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["train"])
    config = {}
    config.update(vars(args))
    wandb.config.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if str(device) != "cuda" and os.environ["HOME"] != '/Users/evanracah':
        assert False, "device must be cuda!"

    train_mode = "probe" if args.method == "supervised" else "train_encoder"
    data = get_episodes(steps=args.num_frames,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.collect_mode,
                                 train_mode=train_mode,
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)

    print_memory("Encoder Eps Loaded")
    torch.set_num_threads(1)
    if args.method == "supervised":
        from src.baselines.slot_supervised import SupervisedTrainer
        tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = data
        trainer = SupervisedTrainer(args, device=device, wandb=wandb)
        encoder = trainer.train(tr_eps, tr_labels, val_eps, val_labels)
        test_acc, test_acc_slots = trainer.test(test_eps, test_labels)
        test_acc_slots = append_suffix(test_acc_slots, "_supervised_test_acc")
        wandb.run.summary.update({"supervised_overall_test_acc": test_acc})
        wandb.run.summary.update(test_acc_slots)

    else:
        tr_eps, val_eps = data
        if args.method == "scn":
            from src.scn import SCNTrainer
            trainer = SCNTrainer(args, device=device, wandb=wandb, ablation=args.ablation)

        if args.method == "cswm":
            pass

        if args.method == "stdim":
            pass
        encoder = trainer.train(tr_eps, val_eps)


    torch.save(encoder.cpu().state_dict(), wandb.run.dir + "/encoder.pt")
    wrd = Path(wandb.run.dir)
    fd = Path(args.final_dir)
    shutil.copytree(wrd.absolute(), fd.absolute() / wrd.stem)
