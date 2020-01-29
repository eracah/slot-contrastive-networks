import torch
from torch.utils.data import DataLoader
import numpy as np

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
        #action = self.experience_buffer[ep]['action'][step]
        next_obs = self.episodes[ep][step + 1] / 255.

        return obs, next_obs
