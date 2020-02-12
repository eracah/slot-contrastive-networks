import argparse
import os
import torch
import wandb
from src.encoders import EncoderCNNMedium, NaureCNNObjExtractor, NatureCNN, EncoderMLP
from src.data.stdim_dataloader import get_stdim_dataloader
from src.data.cswm_dataloader import get_cswm_dataloader
import numpy as np
import torch.nn as nn
from src import cswm_utils
import logging
import gym
# methods that need encoder trained before
ablations = ["nce", "hybrid", "loss1-only", "loss2-only", "none"]
baselines = ["supervised", "random-cnn", "stdim", "cswm"]


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="./temp")
    parser.add_argument('--num-frames', type=int, default=100000,  help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo", "cswm"], default="random_agent")
    parser.add_argument('--num-processes', type=int, default=8, help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4', help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=1, help='Number of frames to stack for a state')
    parser.add_argument("--screen-size", nargs=2, type=int, default=(210, 160))
    parser.add_argument("--downsample", default=False, dest="screen_size", action='store_const', const=(84, 84))
    parser.add_argument("--frameskip",type=int, default=4)
    parser.add_argument("--grayscale", action='store_true', default=False)
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", action="store_false", dest="grayscale")
    parser.add_argument("--checkpoint-index", type=int, default=-1)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument('--method', type=str, default='scn', choices= baselines + ["scn"], help='Method to use for training representations (default: scn')
    parser.add_argument('--ablation', type=str, default="none", choices=ablations, help='Ablation of scn (default: scn')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Dimensionality of embedding.')
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-episode-steps", type=int, default=-1)
    parser.add_argument("--warmstart", type=int, default=0)
    parser.add_argument("--crop",nargs=2,type=int, default=[-1,-1])
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate for learning representations (default: 5e-4)')
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
    # parser.add_argument('--seed', type=int, default=42,
    #                     help='Random seed (default: 42).')
    # parser.add_argument('--name', type=str, default='none',
    #                     help='Experiment name.')
    # parser.add_argument('--save-folder', type=str,
    #                     default='checkpoints',
    #                     help='Path to checkpoints.')
    return parser

def do_epoch(loader, optimizer, model, epoch):
    total_loss = 0.
    total_acc = 0.
    for batch_idx, data_batch in enumerate(loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        loss = model.calc_loss(*data_batch)

        if model.training:
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(tr_loader.dataset),
                               100. * batch_idx / len(loader),
                               loss.item() / len(data_batch[0])))

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss



def get_encoder(args, sample_frame):
    input_channels = sample_frame.shape[1]
    if args.regime == "stdim":
        if args.method == "stdim":
            encoder = NatureCNN(input_channels, args.embedding_dim)
        else:
            obj_extractor = NaureCNNObjExtractor(input_channels, args.num_slots)
            slot_mlp = EncoderMLP(input_dim=np.prod(obj_extractor.final_fmap_shape),
                                  output_dim=args.embedding_dim,
                                  hidden_dim=args.hidden_dim,
                                  num_objects=args.num_slots)
            encoder = nn.Sequential(obj_extractor, slot_mlp)
    elif args.regime == "cswm":
        width_height = np.asarray(sample_frame.shape[2:])
        obj_extractor = EncoderCNNMedium(input_dim=input_channels,
                                        hidden_dim=args.hidden_dim // 16,
                                        num_objects=args.num_slots)
        slot_mlp = EncoderMLP(input_dim=np.prod(width_height // 5),
                                  output_dim=args.embedding_dim,
                                  hidden_dim=args.hidden_dim,
                                  num_objects=args.num_slots)
        encoder = nn.Sequential(obj_extractor, slot_mlp)
    else:
        assert False
    return encoder



if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    wandb.init(project=args.wandb_proj, dir=args.run_dir, tags=["train"])
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    log_file = os.path.join(wandb.run.dir, 'log.txt')
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    logger = logging.getLogger()
    #logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.warning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if str(device) != "cuda" and os.environ["HOME"] != '/Users/evanracah':
        assert False, "device must be cuda!"

    if args.regime == "stdim":
        dataloaders = get_stdim_dataloader(args, mode="eval" if args.method == "supervised" else "train")
    elif args.regime == "cswm":
        dataloaders = get_cswm_dataloader(args,mode="eval" if args.method == "supervised" else "train")
    else:
        assert False

    sample_frame = next(dataloaders[0].__iter__())[0]
    encoder = get_encoder(args, sample_frame)


    if args.method == "supervised":
        from atariari.benchmark.probe import ProbeTrainer
        tr_dl, val_dl, test_dl, label_keys = dataloaders
        sample_label = next(tr_dl.__iter__())[-1]
        num_state_variables = sample_label.shape[1]
        trainer = ProbeTrainer(encoder=encoder,
                               epochs=args.epochs,
                               lr =args.lr,
                               num_state_variables= num_state_variables,
                               patience=args.patience,
                               batch_size=args.batch_size,
                               fully_supervised=True,
                               representation_len=args.embedding_dim)
        trainer.train(tr_dl, val_dl)
        test_acc, test_f1 = trainer.test(test_dl)
        torch.save(encoder.state_dict(), wandb.run.dir + "/encoder.pt")

    else:
        tr_loader, val_loader = dataloaders

        if args.method == "cswm":
            action_dim = gym.make(args.env_name).action_space.n
            from src.baselines.cswm import ContrastiveSWM
            model = ContrastiveSWM(
                encoder=encoder,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                action_dim=action_dim,
                num_objects=args.num_slots,
                sigma=args.sigma,
                hinge=args.hinge,
                ignore_action=args.ignore_action,
                copy_action=args.copy_action).to(device)

            model.apply(cswm_utils.weights_init)

        elif args.method == "scn":
            from src.scn import SCNModel
            model = SCNModel(args, encoder,  device=device, wandb=wandb, ablation=args.ablation).to(device)
            if args.regime == "cswm":
                model.apply(cswm_utils.weights_init)

        elif args.method == "stdim":
            from src.baselines.stdim import STDIMModel
            model = STDIMModel(encoder, args.embedding_dim, config, device, wandb).to(device)
        else:
            assert False


        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr)

        # Train model.
        print('Starting model training...')
        step = 0
        best_loss = 1e9

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0
            tr_loss = do_epoch(tr_loader, optimizer, model, epoch)
            wandb.log({"tr_epoch_loss": tr_loss})
            print('====> Epoch: {} Train average loss: {:.6f}'.format(
                epoch, tr_loss))

            model.eval()
            val_loss = do_epoch(val_loader, optimizer, model, epoch)
            wandb.log({"val_epoch_loss": val_loss})
            print('====> \t Val average loss: {:.6f}'.format(
                val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.encoder.state_dict(), wandb.run.dir + "/encoder.pt")

