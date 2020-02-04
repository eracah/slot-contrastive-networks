import torch
from torch.utils.data import DataLoader
import numpy as np
from atariari.benchmark.episodes import get_episodes
from torch.utils.data import DataLoader, TensorDataset
class EpisodeDataset(torch.utils.data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, episodes):
        self.episodes = episodes
        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = []
        step = 0
        for ep in range(len(episodes)):
            num_steps = episodes[ep].shape[0]
            idx_tuple = [(ep, idx) for idx in range(num_steps - 1)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return len(self.idx2episode)

    def __getitem__(self, idx):

        ep, step = self.idx2episode[idx]


        obs = self.episodes[ep][step] / 255.
        action = -1.
        next_obs = self.episodes[ep][step + 1] / 255.

        return obs, action, next_obs


def get_stdim_dataloader(args,mode="train"):
    if mode == "train":
        return get_stdim_train_dataloader(args)
    else:
        return get_stdim_eval_dataloader(args)


def get_stdim_train_dataloader(args):
    data = get_episodes(steps=args.num_frames,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.collect_mode,
                                 train_mode="train_encoder",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)
    tr_eps, val_eps = data
    tr_dataset = EpisodeDataset(tr_eps)
    tr_dl = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = EpisodeDataset(val_eps)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return tr_dl, val_dl

def make_labelled_dataloader(eps,label_dict, batch_size):
    labels = torch.tensor(list(label_dict.values())).long()
    labels_tensor = labels.transpose(1, 0)
    ds = TensorDataset(eps, labels_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl


def get_stdim_eval_dataloader(args):
    data = get_episodes(steps=args.num_frames,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.collect_mode,
                                 train_mode="probe",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = data
    tr_dl = make_labelled_dataloader(tr_eps,tr_labels,args.batch_size)
    val_dl = make_labelled_dataloader(val_eps, val_labels, args.batch_size)
    test_dl = make_labelled_dataloader(test_eps, test_labels, args.batch_size)

    return tr_dl, val_dl, test_dl
