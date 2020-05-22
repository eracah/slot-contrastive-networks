import argparse
import wandb
import numpy as np
from src.evaluation.metrics import compute_dci_disentangling
from src.utils import postprocess_and_log_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".")
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument("--id", type=str)
    args = parser.parse_args()

    api = wandb.Api()
    path = "/".join(["eracah", args.wandb_proj, args.id])
    print(path)
    run = api.run(path=path)
    #json_obj = run.file(name="wandb-metadata.json").download(root=wandb.run.dir + "/train_run", replace=True)
    #eval_args = json.load(open(json_obj.name))["args"]
    eval_args = run.config

    label_keys = run.summary["label_keys"]
    r2_score = run.summary["concat_overall_localization_avg_r2_lin_reg"]

    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["disentangle"])
    wandb.run.summary.update(dict(concat_overall_localization_avg_r2=r2_score))


    new_args = {}
    for k, v in eval_args.items():
        if k in args:
            args.__dict__["eval_" + k] = v
        else:
            new_args[k] = v


    wandb.config.update(vars(args))
    wandb.config.update(new_args)


    for probe_model in ["lin_reg", "gbt"]:
        file_obj = run.file(name=probe_model + "_probe_weights.npy").download(root=wandb.run.dir, replace=True)
        weights_path = file_obj.name


        weights = np.load(weights_path)
        # compute dci_d and dci_c
        dci_d, dci_c = compute_dci_disentangling(weights, label_keys,
                                                 eval_args["num_slots"], normalize=probe_model == "lin_reg")

        # np.save(wandb.run.dir + "/" + probe_model + "_probe_weights.npy", weights)
        # postprocess_and_log_metrics(dict(zip(label_keys, score)), prefix="concat_",
        #                             suffix="_r2_" + probe_model)
        wandb.run.summary.update({"dci_c_" + probe_model: dci_c, "dci_d_" + probe_model: dci_d})

    print("done")
