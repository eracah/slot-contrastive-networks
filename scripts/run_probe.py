from scripts.run_encoder import train_encoder, train_supervised_encoder
from src.future import  SKLearnProbeTrainer, get_feature_vectors
import torch
from src.utils import get_argparser, train_encoder_methods, probe_only_methods, prepend_prefix, append_suffix
from src.encoders import SlotIWrapper, SlotEncoder,ConcatenateWrapper
import wandb
import sys
from src.majority import majority_baseline
from atariari.benchmark.episodes import get_episodes
from atariari.benchmark.probe import postprocess_raw_metrics
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
from torch.utils.data import RandomSampler, BatchSampler
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torch import nn
import PIL

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
        #log_fmaps(encoder, test_eps)
        f1s, accs = compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels)
        f1_df, acc_df = pd.DataFrame(f1s), pd.DataFrame(accs)
        compute_assigned_slot_metrics(f1_df, acc_df)
        compute_disentangling(f1_df, acc_df)
        compute_all_slots_metrics(encoder, tr_eps, tr_labels,test_eps, test_labels)


#
# def log_fmaps(encoder, episodes):
#     batch_size = 8
#     indices = np.random.randint(len(episodes), size=(batch_size,))
#
#
#
#     episodes_batch = [episodes[i] for i in indices]
#
#     xs = []
#     for ep_ind, episode in enumerate(episodes_batch):
#         # Get one sample from this episode
#         t = np.random.randint(len(episode))
#         xs.append(episode[t])
#
#     xs = torch.stack(xs) / 255.
#     fmaps, slot_fmaps = encoder.get_fmaps(xs)
#     slot_fmaps = slot_fmaps.detach()
#     fm_upsample = nn.functional.interpolate(slot_fmaps, size=xs.shape[-2:], mode="bilinear")
#     fms = fm_upsample.shape
#     fmu = fm_upsample.reshape(fms[0] * fms[1], 1, *fms[2:])
#     fgrid = make_grid(fmu, nrow=8, padding=0).detach().numpy().transpose(1,2,0)
#     x_repeat = xs.repeat(1, 8, 1, 1).numpy().reshape(64,1,210,160)
#     xgrid = make_grid(torch.tensor(x_repeat), nrow=8, padding=0).detach().numpy().transpose(1,2,0)
#     #fig = plt.figure(1, frameon=False, figsize=(50, 50))
#     im1 = plt.imshow(xgrid[:,:,0], cmap=plt.cm.jet)
#     im2 = plt.imshow(fgrid[:,:,0], cmap=plt.cm.jet, alpha=0.7)
#     plt.axis("off")
#     #wandb.log({"chart": plt})
#     plt.savefig("im.jpg")
#     im = plt.imread("im.jpg")
#     # wandb.log({"chart": plt})
#     wandb.log({"slot_fmaps": [wandb.Image(im, caption="Label")]})


# compute all slots
def compute_all_slots_metrics(encoder, tr_eps, tr_labels,test_eps, test_labels):


    cat_slot_enc = ConcatenateWrapper(encoder)
    f_tr, y_tr = get_feature_vectors(cat_slot_enc, tr_eps, tr_labels)
    f_test, y_test = get_feature_vectors(cat_slot_enc, test_eps, test_labels)
    trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                  lr=args.probe_lr,
                                  patience=args.patience)

    cat_test_acc, cat_test_f1 = trainer.train_test(f_tr, y_tr, f_test, y_test)
    cat_test_acc, cat_test_f1 = postprocess_raw_metrics(cat_test_acc, cat_test_f1)
    cat_test_acc = append_suffix(cat_test_acc, "_all_slots")
    #wandb.run.summary.update(cat_test_acc)
    cat_test_f1 = append_suffix(cat_test_f1, "_all_slots")
    wandb.run.summary.update(cat_test_f1)



# compute slot-wise
def compute_slotwise_metrics(encoder, tr_eps, tr_labels, test_eps, test_labels):
    accs = []
    f1s = []
    for i in range(encoder.num_slots):
        slot_i_enc = SlotIWrapper(encoder,i)
        f_tr, y_tr = get_feature_vectors(slot_i_enc, tr_eps, tr_labels)
        f_test, y_test = get_feature_vectors(slot_i_enc, test_eps, test_labels)
        trainer = SKLearnProbeTrainer(epochs=args.epochs,
                                      lr=args.probe_lr,
                                      patience=args.patience)

        test_acc, test_f1score = trainer.train_test(f_tr, y_tr, f_test, y_test)

        accs.append(deepcopy(test_acc))
        f1s.append(deepcopy(test_f1score))
        sloti_test_acc = append_suffix(test_acc, "_acc_slot{}".format(i+1))
        sloti_test_f1 = append_suffix(test_f1score, "_f1_slot{}".format(i+1))
        #wandb.run.summary.update(sloti_test_acc)
        wandb.run.summary.update(sloti_test_f1)
    return f1s, accs


def compute_assigned_slot_metrics(f1_df,acc_df):
    f1_np = f1_df.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_f1s = {f1_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                      (slot_num, factor_num) in inds}

    acc_np = acc_df.to_numpy()
    row_ind, col_ind = lsa(-acc_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_accs = {acc_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                      (slot_num, factor_num) in inds}

    assigned_slot_accs, assigned_slot_f1s = postprocess_raw_metrics(assigned_slot_accs, assigned_slot_f1s)
    assigned_slot_accs = append_suffix(assigned_slot_accs,"_assigned_slot_acc")
    assigned_slot_f1s = append_suffix(assigned_slot_f1s,"_assigned_slot_f1")
    #wandb.run.summary.update(assigned_slot_accs)
    wandb.run.summary.update(assigned_slot_f1s)



# compute disentangling
def compute_disentangling(df, acc_df):
    saps_compactness = append_suffix(compute_SAP(df), "_f1_sap_compactness")
    wandb.run.summary.update(saps_compactness)
    avg_sap_compactness = np.mean(list(saps_compactness.values()))
    wandb.run.summary.update({"f1_avg_sap_compactness": avg_sap_compactness})
    acc_saps_compactness = append_suffix(compute_SAP(acc_df), "_acc_sap_compactness")
    #wandb.run.summary.update(acc_saps_compactness)
    acc_avg_sap_compactness = np.mean(list(acc_saps_compactness.values()))
    #wandb.run.summary.update({"acc_avg_sap_compactness": acc_avg_sap_compactness})
    f1_maxes = dict(df.max())
    acc_maxes = dict(acc_df.max())
    acc_maxes, f1_maxes = postprocess_raw_metrics(acc_maxes, f1_maxes)
    f1_maxes = {k:v for k, v in f1_maxes.items() if "avg" in k}
    f1_maxes = append_suffix(f1_maxes, "_best_slot_for_each")
    wandb.run.summary.update(f1_maxes)
    #wandb.run.summary.update(acc_maxes)

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

