from src.utils import print_memory
from atariari.benchmark.episodes import get_episodes
from scipy.optimize import linear_sum_assignment as lsa
import pandas as pd
from src.evaluation.probe_modules import AttentionProbeTrainer
from src.utils import log_metrics, postprocess_and_log_metrics
from src.encoders import ConcatenateSlots
import argparse
import torch
import wandb
from src.evaluation.probe_modules import ProbeTrainer
from pathlib import Path
import json
import numpy as np
from scripts.train import get_argparser as get_train_argparser
from scripts.train import get_encoder
import glob
import os
from src.data.stdim_dataloader import get_stdim_dataloader
from src.data.cswm_dataloader import get_cswm_dataloader
from matplotlib.ticker import MultipleLocator, IndexFormatter
from matplotlib import pyplot as plt
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
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for  (default: 100)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--wandb-tr-id", type=str)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--max-episode-steps", type=int, default=-1)
    args = parser.parse_args()

    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["eval"])

    # list_of_files = [f for f in glob.glob(args.train_run_parent_dir + "/*") if os.path.isdir(f) and "run" in os.path.basename(f)]  # * means all if need specific format then *.csv
    # latest_file = max(list_of_files, key=os.path.getctime)
    # args.train_run_dirname = Path(latest_file).name
    api = wandb.Api()
    path = "/".join(["eracah", args.wandb_proj, args.wandb_tr_id])
    print(path)
    run = api.run(path=path)
    json_obj = run.file(name="wandb-metadata.json").download(root=wandb.run.dir + "/train_run", replace=True)
    train_args = json.load(open(json_obj.name))["args"]
    file_obj = run.file(name="encoder.pt").download(root=wandb.run.dir, replace=True)
    weights_path = file_obj.name

    train_parser = get_train_argparser()
    train_args = train_parser.parse_args(train_args)

    for k, v in vars(train_args).items():
        if k in args:
            args.__dict__["train_" + k] = v
        else:
            args.__dict__[k] = v

    wandb.config.update(vars(args))

    if args.regime == "stdim":
        dataloaders = get_stdim_dataloader(args, mode="eval")
    elif args.regime == "cswm":
        dataloaders = get_cswm_dataloader(args, mode="eval")

    representation_len = args.embedding_dim if args.method == "stdim" else args.num_slots * args.embedding_dim
    tr_dl, val_dl, test_dl, label_keys = dataloaders
    wandb.run.summary.update(dict(label_keys=label_keys))
    sample_frame = next(tr_dl.__iter__())[0]
    print_memory("after episodes loaded")

    encoder = get_encoder(args, sample_frame)
    print("Loading weights from %s" % weights_path)
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    print_memory("after encoder trained/loaded")
    f1s = []
    if args.method != "stdim":
        encoder = ConcatenateSlots(encoder)
        representation_len = args.num_slots * args.embedding_dim
    else:
        representation_len = args.embedding_dim
    #num_slots = args.num_slots
    #representation_len = args.embedding_dim
    num_slots = 1


    trainer = ProbeTrainer(encoder=encoder,
                           wandb=wandb,
                           epochs=args.epochs,
                           lr=args.lr,
                           patience=args.patience,
                           batch_size=args.batch_size,
                           num_state_variables=len(label_keys),
                           fully_supervised=(args.method == "supervised"),
                           num_slots = num_slots,
                           representation_len=representation_len, l1_regularization=False)


    trainer.train(tr_dl, val_dl)
    test_acc, test_f1score = trainer.test(test_dl)
    weights = trainer.get_weights()
    np.save(wandb.run.dir + "/probe_weights.npy", weights)
    test_f1_dict = dict(zip(label_keys, test_f1score))
    postprocess_and_log_metrics(test_f1_dict, prefix="concat_",
                                suffix="_f1")




    # else:
    #     acc_array = np.asarray(test_acc).reshape(num_state_variables, num_slots).transpose(1, 0)
    #     f1_array = np.asarray(test_f1score).reshape(num_state_variables, num_slots).transpose(1, 0)
    #     f1s = [dict(zip(label_keys, f1)) for f1 in f1_array]
    #
    #     slotwise_expl_df = pd.DataFrame(f1s)
    #     slotwise_expl_dict = {col: slotwise_expl_df.values[:, i] for i, col in enumerate(slotwise_expl_df.columns)}
    #     log_metrics(slotwise_expl_dict, prefix="slotwise_", suffix="_f1")
    #
    #     best_slot_expl = dict(slotwise_expl_df.max())
    #     postprocess_and_log_metrics(best_slot_expl, prefix="best_slot_",
    #                                 suffix="_f1")
    #
    #
    #     f1_np = slotwise_expl_df.to_numpy()
    #     row_ind, col_ind = lsa(-f1_np)
    #     inds = list(zip(row_ind, col_ind))
    #     matched_slot_expl = {slotwise_expl_df.columns[factor_num]: f1_np[slot_num, factor_num] for
    #                          (slot_num, factor_num) in inds}
    #     postprocess_and_log_metrics(matched_slot_expl, prefix="matched_slot_",
    #                                 suffix="_f1")












