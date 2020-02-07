from src.utils import print_memory
from atariari.benchmark.episodes import get_episodes
from scipy.optimize import linear_sum_assignment as lsa
import pandas as pd
from src.evaluation.probe_modules import AttentionProbeTrainer
from src.utils import log_metrics, postprocess_and_log_metrics
import argparse
import torch
import wandb
from atariari.benchmark.probe import ProbeTrainer
from pathlib import Path
import json
import numpy as np
from scripts.train import get_argparser as get_train_argparser
from scripts.train import get_encoder
import glob
import os
from src.data.stdim_dataloader import get_stdim_dataloader
from src.data.cswm_dataloader import get_cswm_dataloader
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument('--num-frames', type=int, default=50000, help='Number of steps to train probes (default: 30000 )')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"],default="random_agent", help="how we collect the data")
    parser.add_argument('--num-processes', type=int, default=8,help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate for learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for  (default: 100)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--train-run-parent-dir", type=str, default=".")
    parser.add_argument("--train-run-dirname", type=str)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--max-episode-steps", type=int, default=-1)
    args = parser.parse_args()



    if args.train_run_dirname is None:
        list_of_files = [f for f in glob.glob(args.train_run_parent_dir + "/*") if os.path.isdir(f) and "run" in os.path.basename(f)]  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        args.train_run_dirname = Path(latest_file).name


    train_run_path = Path(args.train_run_parent_dir) / Path(args.train_run_dirname)
    weights_path = train_run_path / Path("encoder.pt")
    args_path = train_run_path / "wandb-metadata.json"
    train_args = json.load(open(args_path))["args"]

    train_parser = get_train_argparser()
    train_args, unknown = train_parser.parse_known_args(train_args)

    for k, v in vars(train_args).items():
        if k in args:
            args.__dict__["train_" + k] = v
        else:
            args.__dict__[k] = v

    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["eval"])
    wandb.config.update(vars(args))

    if args.regime == "stdim":
        dataloaders = get_stdim_dataloader(args, mode="eval")
    elif args.regime == "cswm":
        dataloaders = get_cswm_dataloader(args, mode="eval")

    tr_dl, val_dl, test_dl, label_keys = dataloaders
    sample_frame = next(tr_dl.__iter__())[0]
    print_memory("after episodes loaded")

    encoder = get_encoder(args, sample_frame)
    print("Loading weights from %s" % weights_path)
    encoder.load_state_dict(torch.load(weights_path))
    encoder.to(device)
    encoder.eval()
    print_memory("after encoder trained/loaded")

    # for probe_type in ["linear", "mlp"]:
    #     attn_probe = AttentionProbeTrainer(encoder=encoder, epochs=args.epochs, lr=args.lr, type=probe_type)
    #     scores, importances_df = attn_probe.train_test(x_tr, x_val, y_tr, y_val, x_test, y_test)
    #     imp_dict = {col: importances_df.values[:, i] for i, col in enumerate(importances_df.columns)}
    #     log_metrics(imp_dict, prefix="%s_attn_probe_"%probe_type, suffix="_weights")
    #     log_metrics(scores, prefix="%s_attn_probe_"%probe_type, suffix="_f1")


    f1s = []
    representation_len = args.embedding_dim
    num_state_variables = len(label_keys)
    num_slots = args.num_slots
    trainer = ProbeTrainer(encoder=encoder,
                           epochs=args.epochs,
                           lr=args.lr,
                           batch_size=args.batch_size,
                           num_state_variables=num_state_variables,
                           fully_supervised=(args.method == "supervised"),
                           representation_len=representation_len)

    trainer.train(tr_dl, val_dl)
    test_acc, test_f1score = trainer.test(test_dl)
    if args.method == "stdim":
        test_f1_dict = dict(zip(label_keys, test_f1score))
        postprocess_and_log_metrics(test_f1_dict, prefix="stdim_",
                                    suffix="_f1")
    else:
        acc_array = np.asarray(test_acc).reshape(num_state_variables, num_slots).transpose(1, 0)
        f1_array = np.asarray(test_f1score).reshape(num_state_variables, num_slots).transpose(1, 0)
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












