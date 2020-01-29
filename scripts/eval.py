from src.utils import print_memory
from src.encoders import SlotEncoder, SlotIWrapper
from atariari.benchmark.episodes import get_episodes
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
import pandas as pd
from src.evaluation.probe_modules import AttentionProbeTrainer
from src.utils import log_metrics, postprocess_and_log_metrics
import argparse
import torch
import wandb
from atariari.benchmark.probe import train_all_probes
from src.utils import get_channels
from pathlib import Path
import json
import numpy as np
from scripts.train import get_argparser as get_train_argparser
import glob
import os
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument('--num-frames', type=int, default=50000,help='Number of steps to train probes (default: 30000 )')
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"],default="random_agent", help="how we collect the data")
    parser.add_argument('--num-processes', type=int, default=8,help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate for learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for  (default: 100)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--train-run-parent-path", type=str, default=".")
    parser.add_argument("--train-run-dir", type=str)
    args = parser.parse_args()



    if args.train_run_dir is None:
        list_of_files = [f for f in glob.glob(args.train_run_parent_path + "/*") if os.path.isdir(f)]  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        args.train_run_dir = Path(latest_file).name

    train_run_parent_path = Path(args.train_run_parent_path)
    train_run_dir = Path(args.train_run_dir)
    train_run_path = train_run_parent_path / train_run_dir
    weights_path = train_run_path / Path("encoder.pt")
    args_path = train_run_path / "wandb-metadata.json"
    train_json_obj = json.load(open(args_path))
    train_args = train_json_obj["args"]

    train_parser = get_train_argparser()
    train_args = train_parser.parse_args(train_args)

    for k, v in vars(train_args).items():
        if k not in args:
            args.__dict__[k] = v


    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["eval"])
    wandb.config.update(vars(args))
    num_channels = get_channels(args)
    encoder = SlotEncoder(input_channels=num_channels, slot_len=args.slot_len, num_slots=args.num_slots, args=args)

    print("Loading weights from %s" % weights_path)
    encoder.load_state_dict(torch.load(weights_path))
    encoder.to(device)
    encoder.eval()
    print_memory("after encoder trained/loaded")

    x_tr, x_val, y_tr, y_val, x_test, y_test = get_episodes(steps=args.num_frames,
                                                             env_name=args.env_name,
                                                             seed=args.seed,
                                                             num_processes=args.num_processes,
                                                             num_frame_stack=args.num_frame_stack,
                                                             downsample=not args.no_downsample,
                                                             color=args.color,
                                                             entropy_threshold=args.entropy_threshold,
                                                             collect_mode=args.collect_mode,
                                                             train_mode="probe",
                                                             checkpoint_index=args.checkpoint_index,
                                                             min_episode_length=args.batch_size
                                                             )
    print_memory("after episodes loaded")



    # for probe_type in ["linear", "mlp"]:
    #     attn_probe = AttentionProbeTrainer(encoder=encoder, epochs=args.epochs, lr=args.lr, type=probe_type)
    #     scores, importances_df = attn_probe.train_test(x_tr, x_val, y_tr, y_val, x_test, y_test)
    #     imp_dict = {col: importances_df.values[:, i] for i, col in enumerate(importances_df.columns)}
    #     log_metrics(imp_dict, prefix="%s_attn_probe_"%probe_type, suffix="_weights")
    #     log_metrics(scores, prefix="%s_attn_probe_"%probe_type, suffix="_f1")


    f1s = []
    representation_len = encoder.slot_len
    num_state_variables = len(y_tr.keys())
    label_keys = y_tr.keys()
    num_slots = args.num_slots
    test_acc, test_f1score = train_all_probes(encoder, x_tr, x_val, x_test, y_tr, y_val, y_test, lr=args.lr,
                                              representation_len=representation_len, args=args, save_dir=wandb.run.dir)
    acc_array = np.asarray(test_acc).reshape(num_state_variables, num_slots).transpose(1,0)
    f1_array = np.asarray(test_f1score).reshape(num_state_variables, num_slots).transpose(1,0)
    f1s = [dict(zip(label_keys, f1)) for f1 in f1_array]

    slotwise_expl_df = pd.DataFrame(f1s)
    slotwise_expl_dict = {col: slotwise_expl_df.values[:, i] for i, col in enumerate(slotwise_expl_df.columns)}
    log_metrics(slotwise_expl_dict, prefix="slotwise_", suffix="_f1")

    best_slot_expl = dict(slotwise_expl_df.max())
    postprocess_and_log_metrics(best_slot_expl, prefix="best_slot_",
                                suffix="_f1")


    f1_np = slotwise_expl_df.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    matched_slot_expl = {slotwise_expl_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                         (slot_num, factor_num) in inds}
    postprocess_and_log_metrics(matched_slot_expl, prefix="matched_slot_",
                                suffix="_f1")












