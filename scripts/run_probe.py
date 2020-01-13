from scripts.run_encoder import train_encoder, train_supervised_encoder
from src.evaluate import get_feature_vectors
import torch
from src.utils import get_argparser, train_encoder_methods
from src.encoders import SlotEncoder
from src.metrics import compute_and_log_raw_quant_metrics
import wandb
from src.majority import majority_baseline
from atariari.benchmark.episodes import get_episodes
from src.visualize import plot_fmaps
import gc
import os
import psutil

def get_probe_feature_vectors(args, encoder):
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss / 10.0 ** 9)  # in bytes
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

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 10.0**9 , flush=True)  # in bytes
    fig = plot_fmaps(encoder, test_eps, num_repeat=encoder.num_slots)
    fig.savefig(wandb.run.dir + "/fmaps.png")
    f_tr, y_tr, f_val, y_val, f_test, y_test = get_feature_vector_tr_split(encoder, tr_eps, tr_labels,val_eps,
                                                                           val_labels,  test_eps, test_labels)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 10.0**9 , flush=True)  # in bytes

    return f_tr, y_tr, f_val, y_val, f_test, y_test



def run_probe(args):
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 10.0**9 , flush=True)  # in bytes
    encoder = get_encoder(args)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 10.0**9 , flush=True)  # in bytes
    f_tr, y_tr, f_val, y_val, f_test, y_test = get_probe_feature_vectors(args, encoder)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 10.0**9 , flush=True)  # in bytes
    compute_and_log_raw_quant_metrics(args, f_tr, y_tr, f_val, y_val,  f_test, y_test)



def get_encoder(args):
    if args.method == "supervised":
        encoder = train_supervised_encoder(args)

    elif args.method == "random-cnn":
        if args.color and args.num_frame_stack == 1:
            num_channels = 3
        elif not args.color:
            num_channels = args.num_frame_stack

        encoder = SlotEncoder(input_channels=num_channels, slot_len=args.slot_len, num_slots=args.num_slots, args=args)

    elif args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()
    else:
        assert False, "No known method specified! Don't know what {} is".format(args.method)

    encoder.cpu()
    return encoder

def get_feature_vector_tr_split(encoder, tr_eps, tr_labels,val_eps, val_labels,  test_eps, test_labels):
    f_tr, y_tr = get_feature_vectors(encoder, tr_eps, tr_labels)
    f_val, y_val = get_feature_vectors(encoder, val_eps, val_labels)
    f_test, y_test = get_feature_vectors(encoder, test_eps, test_labels)
    return f_tr, y_tr, f_val, y_val,  f_test, y_test




if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["probe"]
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    run_probe(args)

