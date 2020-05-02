from src.utils import print_memory
from src.utils import postprocess_and_log_metrics
from src.encoders import ConcatenateSlots
import argparse
import torch
import wandb
from src.evaluation.probe_modules import ProbeTrainer, LinearRegressionProbe
import json
import numpy as np
from scripts.train import get_argparser as get_train_argparser
from scripts.train import get_encoder
from src.data.stdim_dataloader import get_stdim_dataloader
from src.data.cswm_dataloader import get_cswm_dataloader
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import copy
from scipy.stats import entropy
from src.utils import all_localization_keys

from src.evaluation.metrics import calc_slot_importances_from_weights, compute_dci_c, compute_dci_d, select_just_localization_rows



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument("--tr-dir", type=str, default=".")
    parser.add_argument('--num-frames', type=int, default=50000, help='Number of steps to train probes (default: 30000 )')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"],default="random_agent", help="how we collect the data")
    parser.add_argument('--seed', type=int, default=11, help='Random seed to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate for learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-Batch Size (default: 64)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--id", type=str)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--max-episode-steps", type=int, default=-1)
    args = parser.parse_args()

    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["eval"])
    wandb.config.update(vars(args))

    try:
        api = wandb.Api()
        path = "/".join(["eracah", args.wandb_proj, args.id])
        print(path)
        run = api.run(path=path)
        json_obj = run.file(name="wandb-metadata.json").download(root=wandb.run.dir + "/train_run", replace=True)
        train_args = json.load(open(json_obj.name))["args"]
        file_obj = run.file(name="encoder.pt").download(root=wandb.run.dir, replace=True)
        weights_path = file_obj.name
        train_args = run.config
    except:
        path = Path(args.tr_dir)
        print(path)
        json_obj = path / Path("wandb-metadata.json")
        train_args = json.load(open(str(json_obj)))["args"]
        weights_path = path / Path("encoder.pt")
        train_parser = get_train_argparser()
        train_args = train_parser.parse_args(train_args)
        train_args = train_args.__dict__

    for k, v in train_args.items():
        if k in args:
            args.__dict__["train_" + k] = v
        else:
            args.__dict__[k] = v

    wandb.config.update(vars(args))

    if args.regime == "stdim":
        dataloaders = get_stdim_dataloader(args, mode="eval")
    elif args.regime == "cswm":
        dataloaders = get_cswm_dataloader(args, mode="eval")

    tr_dl, val_dl, test_dl, label_keys = dataloaders
    wandb.run.summary.update(dict(label_keys=label_keys))
    sample_frame = next(tr_dl.__iter__())[0]
    print_memory("after episodes loaded")

    method = args.method if "random" not in args.method else args.method.split("random_")[-1]
    encoder = get_encoder(method, args, sample_frame=sample_frame)
    print("Loading weights from %s" % weights_path)
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    print_memory("after encoder trained/loaded")
    f1s = []
    if args.method != "stdim":
        encoder = ConcatenateSlots(encoder)
        representation_len = args.num_slots * args.slot_len
    else:
        representation_len = args.global_vector_len
    num_slots = 1

    trainer = LinearRegressionProbe(encoder)
    k = "r2"
    trainer.train(tr_dl, val_dl)
    val_score = trainer.test(val_dl) # don't use test yet!
    weights = copy.deepcopy(trainer.get_weights())
    np.save(wandb.run.dir + "/" + k + "_probe_weights.npy", weights)
    test_dict = dict(zip(label_keys, val_score))
    postprocess_and_log_metrics(test_dict, prefix="concat_",
                                suffix="_"+ k)

    # compute dci_d and dci_c
    slot_importances = calc_slot_importances_from_weights(weights, args.num_slots)
    slot_imp_localization = select_just_localization_rows(slot_importances, label_keys)
    dci_c = compute_dci_c(slot_imp_localization)
    dci_d = compute_dci_d(slot_imp_localization)
    wandb.run.summary.update(dict(dci_c=dci_c, dci_d=dci_d))























