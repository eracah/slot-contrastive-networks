import argparse
import wandb
import numpy as np
from src.evaluation.metrics import calc_slot_importances_from_weights, compute_dci_c, compute_dci_d, select_just_localization_rows



if __name__ == "__main__":
    # eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--id", type=str)
    args = parser.parse_args()

    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["disentangle"])
    wandb.config.update(vars(args))


    api = wandb.Api()
    path = "/".join(["eracah", args.wandb_proj, args.id])
    print(path)
    run = api.run(path=path)
    #json_obj = run.file(name="wandb-metadata.json").download(root=wandb.run.dir + "/train_run", replace=True)
    #eval_args = json.load(open(json_obj.name))["args"]
    file_obj = run.file(name="r2_probe_weights.npy").download(root=wandb.run.dir, replace=True)
    weights_path = file_obj.name
    eval_args =run.config

    label_keys = run.summary["label_keys"]
    r2_score = run.summary["concat_overall_localization_avg_r2"]
    wandb.run.summary.update(dict(concat_overall_localization_avg_r2=r2_score))

    for k, v in eval_args.items():
        if k in args:
            args.__dict__["eval_" + k] = v
        else:
            args.__dict__[k] = v

    wandb.config.update(vars(args))

    weights = np.load(weights_path)
    # compute dci_d and dci_c
    slot_importances = calc_slot_importances_from_weights(weights, args.num_slots)
    slot_imp_localization = select_just_localization_rows(slot_importances, label_keys)
    dci_c = compute_dci_c(slot_imp_localization)
    dci_d = compute_dci_d(slot_imp_localization)
    wandb.run.summary.update(dict(r2_dci_c=dci_c, r2_dci_d=dci_d))
