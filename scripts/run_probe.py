from scripts.run_encoder import train_encoder, train_supervised_encoder
from src.future import SKLearnProbeTrainer, get_feature_vectors, postprocess_raw_metrics
import torch
from src.utils import get_argparser, train_encoder_methods, probe_only_methods, prepend_prefix, append_suffix
from src.encoders import SlotIWrapper, SlotEncoder,ConcatenateWrapper
import wandb
from src.majority import majority_baseline
from atariari.benchmark.episodes import get_episodes
import pandas as pd
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
import numpy as np
from scipy.stats import entropy

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
        wandb.run.summary.update(test_acc)
        wandb.run.summary.update(test_f1score)


    else:
        if args.method == "supervised":
            encoder = train_supervised_encoder(args)

        elif args.method == "random-cnn":
            observation_shape = tr_eps[0][0].shape
            encoder = SlotEncoder(observation_shape[0], slot_len=args.slot_len, num_slots=args.num_slots, args=args)

        elif args.method in train_encoder_methods:
            if args.train_encoder:
                print("Training encoder from scratch")
                encoder = train_encoder(args)
                encoder.probing = True
                encoder.eval()
            elif args.weights_path != "None": #pretrained encoder
                observation_shape = tr_eps[0][0].shape
                encoder = SlotEncoder(observation_shape[0], slot_len=args.slot_len, num_slots=args.num_slots, args=args)
                print("Print loading in encoder weights from probe of type {} from the following path: {}"
                      .format(args.method, args.weights_path))
                encoder.load_state_dict(torch.load(args.weights_path))
                encoder.eval()

            else:
                assert False, "No known method specified!"
        else:
            assert False, "No known method specified! Don't know what {} is".format(args.method)

        encoder.cpu()
        tr_eps.extend(val_eps)
        tr_labels.extend(val_labels)
        weights = compute_all_slots_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels)
        compute_dci(encoder, weights)
        #log_fmaps(encoder, test_eps)
        f1s = compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels)
        f1_df = pd.DataFrame(f1s)
        compute_assigned_slot_metrics(f1_df)
        compute_disentangling(f1_df)


def compute_dci(encoder, weights):
    var_names = list(weights.keys())
    importances = []
    for k in var_names:
        raw_imp = np.abs(weights[k])
        normalized_imp = raw_imp / raw_imp.sum(axis=0)
        class_avged_imp = normalized_imp.mean(axis=1)
        importances.append(class_avged_imp)
    imp = np.stack(importances)
    # split into importances for the weights for each slot and sum to get importances for each slot
    slot_importances = imp.reshape(len(var_names), encoder.num_slots, encoder.slot_len).sum(axis=2)
    dci_disentangling = 1 - entropy(slot_importances, base=len(var_names))
    dci_completeness = 1 - entropy(slot_importances.T, base=encoder.num_slots)
    prefixes = ["slot_{}_dci_disentangling".format(i) for i in range(encoder.num_slots)]
    dci_disentangling = dict(zip(prefixes, dci_disentangling))
    dci_completeness = append_suffix(dict(zip(var_names, dci_completeness)),"_dci_completeness")
    wandb.run.summary.update(dci_disentangling)
    wandb.run.summary.update({"avg_dci_disentangling":np.mean(list(dci_disentangling.values()))})
    wandb.run.summary.update(dci_completeness)
    wandb.run.summary.update(({"avg_dci_completeness":np.mean(list(dci_completeness.values()))}))








# compute all slots
def compute_all_slots_metrics(encoder, tr_eps, tr_labels,test_eps, test_labels):


    cat_slot_enc = ConcatenateWrapper(encoder)
    f_tr, y_tr = get_feature_vectors(cat_slot_enc, tr_eps, tr_labels)
    f_test, y_test = get_feature_vectors(cat_slot_enc, test_eps, test_labels)
    trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                  lr=args.probe_lr,
                                  patience=args.patience)

    cat_test_f1, weights = trainer.train_test(f_tr, y_tr, f_test, y_test)
    cat_test_f1 = postprocess_raw_metrics(cat_test_f1)
    cat_test_f1 = append_suffix(cat_test_f1, "_f1_all_slots")
    wandb.run.summary.update(cat_test_f1)
    return weights



# compute slot-wise
def compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels):
    f1s = []
    for i in range(encoder.num_slots):
        slot_i_enc = SlotIWrapper(encoder,i)
        f_tr, y_tr = get_feature_vectors(slot_i_enc, tr_eps, tr_labels)
        f_test, y_test = get_feature_vectors(slot_i_enc, test_eps, test_labels)
        trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                      lr=args.probe_lr,
                                      patience=args.patience)

        test_f1score, weights = trainer.train_test(f_tr, y_tr, f_test, y_test)

        f1s.append(deepcopy(test_f1score))
        sloti_test_f1 = append_suffix(test_f1score, "_f1_slot{}".format(i+1))
        wandb.run.summary.update(sloti_test_f1)
    return f1s


def compute_assigned_slot_metrics(f1_df):
    f1_np = f1_df.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_f1s = {f1_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                      (slot_num, factor_num) in inds}

    assigned_slot_f1s = postprocess_raw_metrics(assigned_slot_f1s)
    assigned_slot_f1s = append_suffix(assigned_slot_f1s,"_f1_assigned_slot")
    wandb.run.summary.update(assigned_slot_f1s)



# compute disentangling
def compute_disentangling(df):
    saps_compactness = append_suffix(compute_SAP(df), "_f1_sap_compactness")
    wandb.run.summary.update(saps_compactness)
    avg_sap_compactness = np.mean(list(saps_compactness.values()))
    wandb.run.summary.update({"avg_f1_sap_compactness": avg_sap_compactness})
    f1_maxes = dict(df.max())
    f1_maxes = postprocess_raw_metrics(f1_maxes)
    f1_maxes = {k:v for k, v in f1_maxes.items() if "avg" in k}
    f1_maxes = append_suffix(f1_maxes, "_f1_best_slot_for_each")
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

