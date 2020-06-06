import argparse
import torch
import wandb
from src.encoders import STDIMEncoder, CSWMEncoder, SlotSTDIMEncoder
from src.baselines.slot_supervised import SupervisedModel
from src.baselines.slot_stdim import SlotSTDIMModel
from src.baselines.stdim import STDIMModel
from src.baselines.cswm import ContrastiveSWM
import numpy as np
from src import cswm_utils
import gym
import os
from src.data.dataloader import get_dataloaders
from src.utils import get_num_objects, get_sample_frame

# methods that need encoder trained before
losses = ["hcn", "smcn", "scn", "sdl", "smdl"]
baselines = ["supervised", "random-cnn", "stdim", "cswm"]
methods = ["slot-stdim"]

def get_args():
    parser = get_argparser()
    args = parser.parse_args()
    return args

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="./temp")
    parser.add_argument('--num-frames', type=int, default=100000,
                        help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"],
                        default="random_agent")
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4',
                        help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=1, help='Number of frames to stack for a state')
    parser.add_argument("--screen-size", nargs=2, type=int, default=(210, 160))
    parser.add_argument("--downsample", default=False, dest="screen_size", action='store_const', const=(84, 84))
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--grayscale", action='store_true', default=False)
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", action="store_false", dest="grayscale")
    parser.add_argument("--checkpoint-index", type=int, default=-1)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument('--method', type=str, default='scn', choices=baselines + methods,
                        help='Method to use for training representations (default: scn')
    parser.add_argument('--losses', nargs="+", type=str, choices=losses,
                        help='Ablation of scn (default: scn')
    parser.add_argument('--global-vector-len', type=int, default=256, help='Dimensionality of embedding.')
    parser.add_argument("--slot-len", type=int, default=32)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=-1)
    parser.add_argument("--warmstart", type=int, default=0)
    parser.add_argument("--crop", nargs=2, type=int, default=[-1, -1])
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning Rate for learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for  (default: 100)')
    parser.add_argument("--wandb-proj", type=str, default="coors-scratch")
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--noop-max', type=int, default=30)
    parser.add_argument("--regime", type=str, default="stdim", choices=["stdim", "cswm"],
                        help="whether to use the encoder and dataloader from stdim or from cswm")
    parser.add_argument('--hidden-dim', type=int, default=512, help='Number of hidden units in transition MLP.')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='How many batches to wait before logging'
                             'training status.')
    parser.add_argument('--action-dim', type=int, default=4,
                        help='Dimensionality of action space.')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Energy scale.')
    parser.add_argument('--hinge', type=float, default=1.,
                        help='Hinge threshold parameter.')
    parser.add_argument('--ignore-action', action='store_true', default=False,
                        help='Ignore action in GNN transition model.')
    parser.add_argument('--copy-action', action='store_true', default=False,
                        help='Apply same action to all object slots.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--regression', action='store_true', default=False,
                        help='whether to use regression for supervised')
    parser.add_argument('--random-cnn', action='store_true', default=False,
                        help='whether to not learn at all and just save random weights')
    return parser


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) != "cuda" and os.environ["HOME"] != '/Users/evanracah':
        assert False, "device must be cuda!"
    return device


def init_wandb(args):
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["train"])
    wandb.config.update(vars(args))
    with open("wandb_id.txt", "w") as f:
        f.write(str(wandb.run.id))


def get_encoder(args, sample_frame):
    input_channels = sample_frame.shape[1]
    width_height = np.asarray(sample_frame.shape[2:])
    if args.method == "stdim":
        encoder = STDIMEncoder(input_channels,
                               global_vector_len=args.num_slots*args.slot_len)
    elif args.method in ["slot-stdim", "supervised", "random-cnn", "cswm"]:
        encoder = SlotSTDIMEncoder(input_channels,
                                   num_slots=args.num_slots,
                                   slot_len=args.slot_len)
    # elif args.method in ["cswm"]:
    #     encoder = CSWMEncoder(input_dim=input_channels,
    #                           hidden_dim=args.hidden_dim // 16,
    #                           num_objects=args.num_slots,
    #                           output_dim=args.slot_len,
    #                           width_height=width_height)
    else:
        assert False, "I don't recognize the method name: {}!".format(args.method)

    return encoder

def get_model(encoder, args, label_keys):
    if args.method == "cswm":
        action_dim = gym.make(args.env_name).action_space.n
        model = ContrastiveSWM(
            encoder=encoder,
            embedding_dim=args.slot_len,
            hidden_dim=args.hidden_dim,
            action_dim=action_dim,
            num_objects=args.num_slots,
            sigma=args.sigma,
            hinge=args.hinge,
            ignore_action=args.ignore_action,
            copy_action=args.copy_action
        ).to(device)

        model.apply(cswm_utils.weights_init)


    elif args.method == "stdim":
        model = STDIMModel(encoder, args, args.slot_len*args.num_slots, device, wandb).to(device)

    elif args.method == "slot-stdim":
        model = SlotSTDIMModel(encoder, args, device, wandb).to(device)

    elif args.method == "supervised":
        model = SupervisedModel(encoder, args, label_keys, wandb=wandb).to(device)

    else:
        assert False
    return model


def do_training(model, tr_loader, val_loader):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr)

    print('Starting model training...')
    best_loss = 1e9

    for epoch in range(args.epochs):
        model.train()
        tr_loss = do_epoch(tr_loader, optimizer, model, epoch)
        print('====> Epoch: {} Train average loss: {:.6f}'.format(
            epoch + 1, tr_loss))

        model.eval()
        val_loss = do_epoch(val_loader, optimizer, model, epoch)
        print('====> \t Val average loss: {:.6f}'.format(
            val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.encoder.state_dict(), wandb.run.dir + "/encoder.pt")


def do_epoch(loader, optimizer, model, epoch):
    total_loss = 0.
    for batch_idx, data_batch in enumerate(loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        loss = model.calc_loss(*data_batch)

        if model.training:
            wandb.log(dict(tr_loss=loss))
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(loader.dataset),
                               100. * batch_idx / len(loader),
                               loss.item() / len(data_batch[0])))

        else:
            wandb.log(dict(val_loss=loss))

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss



if __name__ == "__main__":
    args = get_args()
    device = get_device()

    (tr_dl, val_dl), label_keys = get_dataloaders(args,
                                                  keep_as_episodes=(args.method != "supervised"),
                                                  test_set=False,
                                                  label_keys=True)
    num_objects = get_num_objects(label_keys)
    args.num_slots = num_objects  # we cheat a lil bit here!
    init_wandb(args)

    sample_frame = get_sample_frame(tr_dl)
    encoder = get_encoder(args, sample_frame)
    if args.method == "random-cnn":
        torch.save(encoder.state_dict(), wandb.run.dir + "/encoder.pt")
    else:
        model = get_model(encoder, args, label_keys)
        do_training(model, tr_dl, val_dl)
