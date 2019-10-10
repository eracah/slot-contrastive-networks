from scripts.run_contrastive import train_encoder
from src.future import  SKLearnProbeTrainer, get_feature_vectors
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
        tr_eps.extend(val_eps)
        tr_labels.extend(val_labels)
        encoder.cpu()
        cat_slot_enc = ConcatenateWrapper(encoder)

        f_tr, y_tr = get_feature_vectors(cat_slot_enc, tr_eps, tr_labels)
        f_test, y_test = get_feature_vectors(cat_slot_enc, test_eps, test_labels)
        trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                      lr=args.probe_lr,
                                      patience=args.patience)

        cat_test_acc, cat_test_f1 = trainer.train_test(f_tr, y_tr, f_test, y_test)

        cat_test_acc = prepend_prefix(cat_test_acc, "all_slots_")
        wandb.run.summary.update(cat_test_acc)
        cat_test_f1 = prepend_prefix(cat_test_f1, "all_slots_")
        wandb.run.summary.update(cat_test_f1)



        accs = []
        f1s = []
        for i in range(args.num_slots):
            slot_i_enc = SlotIWrapper(encoder,i)
            f_tr, y_tr = get_feature_vectors(slot_i_enc, tr_eps, tr_labels)
            f_test, y_test = get_feature_vectors(slot_i_enc, test_eps, test_labels)
            trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                          lr=args.probe_lr,
                                          patience=args.patience)

            test_acc, test_f1score = trainer.train_test(f_tr, y_tr, f_test, y_test)

            accs.append(deepcopy(test_acc))
            f1s.append(deepcopy(test_f1score))
            sloti_test_acc = prepend_prefix(test_acc, "slot{}_".format(i+1))
            sloti_test_f1 = prepend_prefix(test_f1score, "slot{}_".format(i+1))
            wandb.run.summary.update(sloti_test_acc)
            wandb.run.summary.update(sloti_test_f1)


    for metrics in [accs, f1s]:
        df = pd.DataFrame(metrics)
        df = df[[c for c in df.columns if "avg" not in c]]
        saps_compactness = prepend_prefix(compute_SAP(df), "SAP_Compactness_")
        wandb.run.summary.update(saps_compactness)
        avg_sap_compactness = np.mean(list(saps_compactness.values()))
        wandb.run.summary.update({"avg_sap_compactness": avg_sap_compactness})

        saps_modularity = prepend_prefix(compute_SAP(df.T), "SAP_Modularity_")
        avg_sap_modularity = np.mean(list(saps_modularity.values()))
        wandb.run.summary.update(saps_modularity)
        wandb.run.summary.update({"avg_sap_modularity": avg_sap_modularity})



        maxes = prepend_prefix(dict(df.max()),"best_slot_")
        argmaxes = prepend_prefix(dict(df.idxmax()), "slot_index_for_best_")
        wandb.run.summary.update(maxes)
        wandb.run.summary.update(argmaxes)





def compute_SAP(df):

    return {str(k): np.abs(df.nlargest(2, [k])[k].diff().iloc[1]) for k in df.columns}



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

