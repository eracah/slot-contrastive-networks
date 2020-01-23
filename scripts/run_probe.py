from scripts.run_encoder import train_encoder, train_supervised_encoder
import torch
from src.utils import get_argparser, train_encoder_methods, print_memory
from src.encoders import SlotEncoder, SlotIWrapper
from atariari.benchmark.episodes import get_episodes
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
import wandb
import pandas as pd
import numpy as np
from scipy.stats import entropy
from src.evaluate import AttentionProbeTrainer
from src.utils import log_metrics, postprocess_and_log_metrics

from atariari.benchmark.probe import train_all_probes

def get_encoder(args):
    if args.color and args.num_frame_stack == 1:
        num_channels = 3
    elif not args.color:
        num_channels = args.num_frame_stack

    if args.method == "supervised":
        encoder = train_supervised_encoder(args)

    elif args.method == "random-cnn":
        encoder = SlotEncoder(input_channels=num_channels, slot_len=args.slot_len, num_slots=args.num_slots, args=args)

    elif args.method in train_encoder_methods:
        if args.weights_path == "None":
            print("Training encoder from scratch")
            encoder = train_encoder(args)
        else:
            print("Loading weights from %s"%args.weights_path)
            encoder = SlotEncoder(input_channels=num_channels, slot_len=args.slot_len, num_slots=args.num_slots,
                                  args=args)
            encoder.load_state_dict(torch.load(args.weights_path))

    else:
        assert False, "No known method specified! Don't know what {} is".format(args.method)
    print_memory("after encoder trained/loaded")
    encoder.eval()
    return encoder


parser = get_argparser()
args = parser.parse_args()
tags = ["probe"]
wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=tags)
config = {}
config.update(vars(args))
wandb.config.update(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_memory("start memory")

encoder = get_encoder(args)
encoder.to(device)

x_tr, x_val, y_tr, y_val, x_test, y_test = get_episodes(steps=args.probe_num_frames,
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

for type_ in ["linear", "mlp"]:
    attn_probe = AttentionProbeTrainer(encoder=encoder, epochs=args.epochs, patience=args.patience, lr=args.probe_lr, type=type)
    scores, importances_df = attn_probe.train_test(x_tr, x_val, y_tr, y_val, x_test, y_test)
    imp_dict = {col: importances_df.values[:, i] for i, col in enumerate(importances_df.columns)}
    log_metrics(imp_dict, prefix="%s_attn_probe_"%type, suffix="_weights")
    log_metrics(scores, prefix="%sattn_probe_"%type, suffix="_f1")


f1s = []
representation_len = encoder.slot_len
for i in range(args.num_slots):
    slot_i_encoder = SlotIWrapper(encoder, i)
    test_acc, test_f1score = train_all_probes(slot_i_encoder, x_tr, x_val, x_test, y_tr, y_val, y_test, representation_len, args, wandb.run.dir)
    f1s.append(deepcopy(test_f1score))

slotwise_expl_df = pd.DataFrame(f1s)
slotwise_expl_dict = {col: slotwise_expl_df.values[:, i] for i, col in enumerate(slotwise_expl_df.columns)}
log_metrics(slotwise_expl_dict, prefix="slotwise_", suffix="_f1")

best_slot_expl = dict(slotwise_expl_df.max())
postprocess_and_log_metrics(best_slot_expl, prefix="best_slot_",
                            suffix="_f1")


f1_np = slotwise_expl_df.to_numpy()
row_ind, col_ind = lsa(-f1_np)
inds = list(zip(row_ind, col_ind))
matched_slot_expl = {slotwise_expl_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                     (slot_num, factor_num) in inds}
postprocess_and_log_metrics(matched_slot_expl, prefix="matched_slot_",
                            suffix="_f1")












