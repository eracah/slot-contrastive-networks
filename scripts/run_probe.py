from scripts.run_encoder import train_encoder
from src.future import  SKLearnProbeTrainer, get_feature_vectors
import torch
from src.utils import get_argparser, train_encoder_methods, probe_only_methods, prepend_prefix, append_suffix
from src.encoders import NatureCNN, SlotIWrapper, SlotEncoder,ConcatenateWrapper
import wandb
import sys
from src.majority import majority_baseline
from atariari.benchmark.episodes import get_episodes
from atariari.benchmark.probe import postprocess_raw_metrics
import pandas as pd
import numpy as np
from copy import deepcopy


def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.method  == "majority":
        test_acc, test_f1score = majority_baseline(tr_labels, test_labels, wandb)

    elif args.method == "supervised":
        observation_shape = tr_eps[0][0].shape
        args.obs_space = observation_shape
        from src.slot_supervised import SupervisedTrainer
        trainer = SupervisedTrainer(args, device=device, wandb=wandb)
        encoder = trainer.train(tr_eps, tr_labels, val_eps, val_labels)
        test_acc, test_acc_slots = trainer.test(test_eps, test_labels)
        test_acc_slots = append_suffix(test_acc_slots, "_supervised_test_acc")
        wandb.run.summary.update({"supervised_overall_test_acc": test_acc})
        wandb.run.summary.update(test_acc_slots)

    else:

        if args.method in train_encoder_methods:
            if args.train_encoder:
                print("Training encoder from scratch")
                encoder = train_encoder(args)
                encoder.probing = True
                encoder.eval()
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



        elif args.method == "random_cnn":
            observation_shape = tr_eps[0][0].shape
            encoder = SlotEncoder(observation_shape[0], args)








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
        cat_test_acc, cat_test_f1 = postprocess_raw_metrics(cat_test_acc, cat_test_f1)
        cat_test_acc = prepend_prefix(cat_test_acc, "all_slots_")
        wandb.run.summary.update(cat_test_acc)
        cat_test_f1 = append_suffix(cat_test_f1, "_all_slots")
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
            sloti_test_f1 = append_suffix(test_f1score, "_f1_slot{}".format(i+1))
            wandb.run.summary.update(sloti_test_acc)
            wandb.run.summary.update(sloti_test_f1)



    df, acc_df = pd.DataFrame(f1s), pd.DataFrame(accs)
    saps_compactness = append_suffix(compute_SAP(df), "_f1_sap_compactness")
    wandb.run.summary.update(saps_compactness)
    avg_sap_compactness = np.mean(list(saps_compactness.values()))
    wandb.run.summary.update({"f1_avg_sap_compactness": avg_sap_compactness})
    f1_maxes = dict(df.max())
    acc_maxes = dict(acc_df.max())
    acc_maxes, f1_maxes = postprocess_raw_metrics(acc_maxes, f1_maxes)
    f1_maxes = {k:v for k, v in f1_maxes.items() if "avg" in k}
    f1_maxes = append_suffix(f1_maxes, "_best_slot_for_each")
    wandb.run.summary.update(f1_maxes)







def compute_variance(df):
    pass

def compute_SAP(df):
    return {str(k): np.abs(df.nlargest(2, [k])[k].diff().iloc[1]) for k in df.columns}






if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["probe"]
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    run_probe(args)

