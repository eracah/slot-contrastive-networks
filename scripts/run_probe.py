from scripts.run_contrastive import train_encoder
from atariari.probe import ProbeTrainer, SKLearnProbeTrainer
import torch
from src.utils import get_argparser, train_encoder_methods, probe_only_methods
from src.encoders import NatureCNN, ImpalaCNN, SlotIWrapper, SlotEncoder,ConcatenateWrapper
import wandb
import sys
from src.majority import majority_baseline
from atariari.episodes import get_episodes
import pandas as pd
import numpy as np
from copy import deepcopy


def run_probe(args):
    #wandb.config.update(vars(args))
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = get_episodes(steps=args.probe_num_frames,
                                                                                 env_name=args.env_name,
                                                                                 seed=args.seed,
                                                                                 num_processes=args.num_processes,
                                                                                 num_frame_stack=args.num_frame_stack,
                                                                                 downsample=not args.no_downsample,
                                                                                 color=args.color,
                                                                                 entropy_threshold=args.entropy_threshold,
                                                                                 collect_mode=args.probe_collect_mode,
                                                                                 train_mode="probe",
                                                                                 checkpoint_index=args.checkpoint_index,
                                                                                 min_episode_length=args.batch_size
                                                                                 )

    print("got episodes!")

    if args.train_encoder and args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()


    elif args.method in ["pretrained-rl-agent", "majority"]:
        encoder = None

    else:
        observation_shape = tr_eps[0][0].shape
        encoder = SlotEncoder(observation_shape[0], args)

        if args.weights_path == "None":
            if args.method not in probe_only_methods:
                sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
        else:
            print("Print loading in encoder weights from probe of type {} from the following path: {}"
                  .format(args.method, args.weights_path))
            encoder.load_state_dict(torch.load(args.weights_path))
            encoder.eval()

    torch.set_num_threads(1)

    if args.method == 'majority':
        test_acc, test_f1score = majority_baseline(tr_labels, test_labels, wandb)

    else:
        cat_slot_enc = ConcatenateWrapper(encoder)
        trainer = SKLearnProbeTrainer(encoder=cat_slot_enc,
                               epochs=args.epochs,
                               lr=args.probe_lr,
                               patience=args.patience)
        cat_test_acc, cat_test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels,
                                                    val_labels, test_eps, test_labels)


        cat_test_acc.update(cat_test_f1score)
        all_metrics = cat_test_acc
        all_metrics = prepend_prefix(all_metrics, "all_slots_")

        accs = []
        f1s = []
        for i in range(args.num_slots):
            slot_i_encoder = SlotIWrapper(encoder,i)
            trainer = SKLearnProbeTrainer(encoder= slot_i_encoder,
                                          epochs=args.epochs,
                                          lr=args.probe_lr,
                                          patience=args.patience)
            test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels,
                                                                val_labels, test_eps, test_labels)
            accs.append(deepcopy(test_acc))
            f1s.append(deepcopy(test_f1score))
            sloti_test_acc = prepend_prefix(test_acc, "slot{}_".format(i))
            sloti_test_f1 = prepend_prefix(test_f1score, "slot{}_".format(i))
            all_metrics.update(sloti_test_acc)
            all_metrics.update(sloti_test_f1)


    for metrics in [accs, f1s]:
        df = pd.DataFrame(metrics)
        df = df[[c for c in df.columns if "avg" not in c]]
        saps = prepend_prefix(compute_SAP(df), "SAP_")
        avg_sap = np.mean(list(saps.values()))
        all_metrics["sap_avg"] = avg_sap
        maxes = prepend_prefix(dict(df.max()),"best_slot_")
        argmaxes = prepend_prefix(dict(df.idxmax()), "slot_index_for_best_")
        all_metrics.update(saps)
        all_metrics.update(maxes)
        all_metrics.update(argmaxes)



    wandb.run.summary.update(all_metrics)


def compute_SAP(df):
    return {k: np.abs(df.nlargest(2, [k])[k].diff().iloc[1]) for k in df.columns}



def prepend_prefix(dictionary, prefix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[prefix + k] = v
    return new_dict


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["probe"]
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    run_probe(args)

