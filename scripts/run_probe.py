from scripts.run_encoder import train_encoder, train_supervised_encoder
from src.evaluate import LinearProbeTrainer, GBTProbeTrainer,\
    MLPProbeTrainer, get_feature_vectors, postprocess_raw_metrics, AttentionProbeTrainer
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
from torch import nn

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

    if args.method == "majority":
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


        wdci_dict = compute_explictness_weighted_dci(encoder, tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels)
        wdci_dict = prepend_prefix(wdci_dict,"wdci_")
        wandb.run.summary.update(wdci_dict)


        tr_eps.extend(val_eps)
        tr_labels.extend(val_labels)
        log_fmaps(encoder, test_eps)

        probe_dict = { "mlp":  MLPProbeTrainer,"gbt":GBTProbeTrainer,"linear":LinearProbeTrainer}
        for probe_name, probe_trainer in probe_dict.items():
            f1s = compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels, probe_trainer=probe_trainer, probe_name=probe_name)
            f1_df = pd.DataFrame(f1s)
            compute_assigned_slot_metrics(f1_df, probe_name)
            compute_disentangling(f1_df, probe_name)
            weights = compute_all_slots_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels,  probe_trainer=probe_trainer, probe_name=probe_name)
            if probe_name == "linear":
                compute_dci(encoder, weights, probe_name)


def log_fmaps(encoder, episodes):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    batch_size = 8
    indices = np.random.randint(len(episodes), size=(batch_size,))



    episodes_batch = [episodes[i] for i in indices]

    xs = []
    for ep_ind, episode in enumerate(episodes_batch):
        # Get one sample from this episode
        t = np.random.randint(len(episode))
        xs.append(episode[t])

    xs = torch.stack(xs) / 255.
    fmaps, slot_fmaps = encoder.get_fmaps(xs)
    slot_fmaps = slot_fmaps.detach()
    fm_upsample = nn.functional.interpolate(slot_fmaps, size=xs.shape[-2:], mode="bilinear")
    fms = fm_upsample.shape
    fmu = fm_upsample.reshape(fms[0] * fms[1], 1, *fms[2:])
    fgrid = make_grid(fmu, nrow=8, padding=0).detach().numpy().transpose(1,2,0)
    x_repeat = xs.repeat(1, 8, 1, 1).numpy().reshape(64,1,210,160)
    xgrid = make_grid(torch.tensor(x_repeat), nrow=8, padding=0).detach().numpy().transpose(1,2,0)
    fig = plt.figure(1, frameon=False, figsize=(50, 50))
    im1 = plt.imshow(xgrid[:,:,0], cmap=plt.cm.jet)
    im2 = plt.imshow(fgrid[:,:,0], cmap=plt.cm.jet, alpha=0.7)
    plt.axis("off")
    plt.savefig(wandb.run.dir + "/im.jpg")
    # wandb.log({"chart": plt})
    #wandb.log({"slot_fmaps": [wandb.Image(im, caption="Label")]})


def compute_explictness_weighted_dci(encoder, tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels):
    f_tr, y_tr = get_feature_vectors(encoder, tr_eps, tr_labels)
    f_val, y_val = get_feature_vectors(encoder, val_eps, val_labels)
    f_test, y_test = get_feature_vectors(encoder, test_eps, test_labels)
    attn_probe = AttentionProbeTrainer(epochs=args.epochs, patience=args.patience)
    f1_dict, weights_dict = attn_probe.train_test(f_tr, y_tr, f_val, y_val, f_test, y_test)
    f1_scores = list(f1_dict.values())
    feat_imp = pd.DataFrame(weights_dict).values
    keys = list(f1_dict.keys())
    base = feat_imp.shape[0]
    dci_disentangling = 1 - entropy(feat_imp, base=base)
    weighted_dci = f1_scores * dci_disentangling
    wdci_dict = dict(zip(keys, weighted_dci))
    wdci_dict = postprocess_raw_metrics(wdci_dict)
    return wdci_dict




def compute_dci(encoder, weights, probe_name="linear"):
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
    prefixes = ["slot_{}_dci_disentangling_{}_probe".format(i, probe_name) for i in range(encoder.num_slots)]
    dci_disentangling = dict(zip(prefixes, dci_disentangling))
    dci_completeness = append_suffix(dict(zip(var_names, dci_completeness)),"_dci_completeness_" + probe_name + "_probe")
    wandb.run.summary.update(dci_disentangling)
    wandb.run.summary.update({"avg_dci_disentangling_" + probe_name + "_probe": np.mean(list(dci_disentangling.values()))})
    wandb.run.summary.update(dci_completeness)
    wandb.run.summary.update(({"avg_dci_completeness_"+ probe_name + "_probe": np.mean(list(dci_completeness.values()))}))


# compute all slots
def compute_all_slots_metrics(encoder, tr_eps, tr_labels,test_eps, test_labels,probe_trainer=LinearProbeTrainer, probe_name="linear"):


    cat_slot_enc = ConcatenateWrapper(encoder)
    f_tr, y_tr = get_feature_vectors(cat_slot_enc, tr_eps, tr_labels)
    f_test, y_test = get_feature_vectors(cat_slot_enc, test_eps, test_labels)
    trainer = probe_trainer(epochs=args.epochs,
                                  lr=args.probe_lr,
                                  patience=args.patience)

    cat_test_f1, weights = trainer.train_test(f_tr, y_tr, f_test, y_test)
    cat_test_f1 = postprocess_raw_metrics(cat_test_f1)
    cat_test_f1 = prepend_prefix(cat_test_f1, "all_slots_f1_" + probe_name + "_probe_")
    wandb.run.summary.update(cat_test_f1)
    return weights



# compute slot-wise
def compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels, probe_trainer=LinearProbeTrainer, probe_name="linear"):
    f1s = []
    for i in range(encoder.num_slots):
        slot_i_enc = SlotIWrapper(encoder,i)
        f_tr, y_tr = get_feature_vectors(slot_i_enc, tr_eps, tr_labels)
        f_test, y_test = get_feature_vectors(slot_i_enc, test_eps, test_labels)
        trainer = probe_trainer(epochs=args.epochs,
                                      lr=args.probe_lr,
                                      patience=args.patience)

        test_f1score, weights = trainer.train_test(f_tr, y_tr, f_test, y_test)

        f1s.append(deepcopy(test_f1score))
        sloti_test_f1 = append_suffix(test_f1score, "_f1_slot{}".format(i+1))
        sloti_test_f1 = append_suffix(sloti_test_f1, "_" + probe_name + "_probe")
        wandb.run.summary.update(sloti_test_f1)
    return f1s


def compute_assigned_slot_metrics(f1_df, probe_name="linear"):
    f1_np = f1_df.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_f1s = {f1_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                      (slot_num, factor_num) in inds}

    assigned_slot_f1s = postprocess_raw_metrics(assigned_slot_f1s)
    assigned_slot_f1s = prepend_prefix(assigned_slot_f1s, "assigned_slot_f1_" + probe_name + "_probe_")
    wandb.run.summary.update(assigned_slot_f1s)



# compute disentangling
def compute_disentangling(df, probe_name="linear"):
    saps_compactness = append_suffix(compute_SAP(df), "_f1_sap_compactness_" + probe_name + '_probe')
    wandb.run.summary.update(saps_compactness)
    avg_sap_compactness = np.mean(list(saps_compactness.values()))
    wandb.run.summary.update({"avg_f1_sap_compactness_" + probe_name + "_probe": avg_sap_compactness})
    f1_maxes = dict(df.max())
    f1_maxes = postprocess_raw_metrics(f1_maxes)
    f1_maxes = {k:v for k, v in f1_maxes.items() if "avg" in k}
    f1_maxes = prepend_prefix(f1_maxes, "best_slot_for_each_f1_" + probe_name + '_probe_')
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

