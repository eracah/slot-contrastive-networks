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
    obs_shape = tr_eps[0][0].shape
    print("got episodes!")

    if args.method == "majority":
        test_acc, test_f1score = majority_baseline(tr_labels, test_labels, wandb)
        #wandb.run.summary.update(test_acc)
        wandb.run.summary.update(test_f1score)
    else:
        encoder = get_encoder(args, observation_shape=obs_shape)
        fig = plot_fmaps(encoder, test_eps, num_repeat=encoder.num_slots)
        fig.savefig(wandb.run.dir + "/fmaps.png")
        f_tr, y_tr, f_val, y_val, f_test, y_test = get_feature_vector_tr_split(encoder, tr_eps, tr_labels,val_eps,
                                                                               val_labels,  test_eps, test_labels)
        compute_and_log_raw_quant_metrics(args, f_tr, y_tr, f_val, y_val,  f_test, y_test)


def get_encoder(args, observation_shape):
    if args.method == "supervised":
        encoder = train_supervised_encoder(args)

    elif args.method == "random-cnn":
        encoder = SlotEncoder(observation_shape[0], slot_len=args.slot_len, num_slots=args.num_slots, args=args)

    elif args.method in train_encoder_methods:
        if args.train_encoder:
            print("Training encoder from scratch")
            encoder = train_encoder(args)
            encoder.probing = True
            encoder.eval()
        elif args.weights_path != "None":  # pretrained encoder
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

