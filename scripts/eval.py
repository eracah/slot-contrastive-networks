from src.utils import print_memory
from src.utils import postprocess_and_log_metrics
from src.encoders import ConcatenateSlots
import argparse
import torch
import wandb
from src.evaluation.probe_modules import GBTRegressionProbe, LinearRegressionProbe
import json
import numpy as np
from scripts.train import get_argparser as get_train_argparser
from scripts.train import get_encoder
from src.data.dataloader import get_dataloaders
from pathlib import Path
import copy
from src.evaluation.metrics import calc_slot_importances_from_weights, compute_dci_c, \
    compute_dci_d, select_just_localization_rows, average_over_obj


def compute_slot_accuracy(encoder, tr_dl, test_dl, probe_model="lin_reg"):
    if probe_model == "lin_reg":
        trainer = LinearRegressionProbe(encoder)
    elif probe_model == "gbt":
        trainer = GBTRegressionProbe(encoder)

    trainer.train(tr_dl)
    test_score = trainer.test(test_dl)
    feature_importances = copy.deepcopy(trainer.get_feature_importances())
    return test_score, feature_importances

def compute_dci_disentangling(feat_imps, label_keys, num_slots, normalize=True):
    if normalize:
        weights = feat_imps
        slot_importances = calc_slot_importances_from_weights(weights, num_slots)
    else:
        slot_importances = feat_imps
    slot_imp_localization, loc_keys = select_just_localization_rows(slot_importances, label_keys)
    obj_importances = average_over_obj(loc_keys, slot_imp_localization)     # select just object rows
    dci_c = compute_dci_c(obj_importances)
    dci_d = compute_dci_d(obj_importances)
    return dci_d, dci_c


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
    parser.add_argument("--probe-model", type=str, default="lin_reg", choices=["lin_reg", "gbt"],
                        help="the model type for probing")
    parser.add_argument("--input-format", default="concat", type=str, choices=["concat","slots"],
                        help="whether to compute slot accuracy by concatenating all\
                             slots or probing each slot separately")
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

    (tr_dl, val_dl, test_dl), label_keys = get_dataloaders(args, keep_as_episodes=False, test_set=True, label_keys=True)
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
    if args.input_format == "concat":
        if args.method != "stdim":
            encoder = ConcatenateSlots(encoder)


    for probe_model in ["lin_reg", "lin_reg"]:
        score, weights = compute_slot_accuracy(encoder,
                                               tr_dl,
                                               test_dl=val_dl,
                                               probe_model=probe_model)  # don't use test yet!
        dci_d, dci_c = compute_dci_disentangling(weights, label_keys,
                                                 args.num_slots, normalize=probe_model == "lin_reg")


        np.save(wandb.run.dir + "/" + probe_model + "_probe_weights.npy", weights)
        postprocess_and_log_metrics(dict(zip(label_keys, score)), prefix="concat_",
                                    suffix="_r2_"+ probe_model)
        wandb.run.summary.update({"dci_c_" + probe_model :dci_c, "dci_d_" + probe_model:dci_d})
