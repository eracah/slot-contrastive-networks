from typing import List, Any

from torch.utils.data import DataLoader
from src.data.wrappers import wrap_atari_env
from atariari.benchmark.wrapper import AtariARIWrapper
import gym
try:
    import wandb
except:
    pass
from src.utils import appendabledict, flatten_labels
import torch
import numpy as np


def get_transitions(args, seed=42, keep_as_episodes=True, min_episode_length=8, max_frames=None, max_episodes=None):
    """workhorse function: collect frames, actions, and labels and split into episodes"""

    rng = np.random.RandomState(seed)
    frames, labels, actions = [], [], []
    env = gym.make(args.env_name)
    env.seed(seed)
    env = wrap_atari_env(env, args, rng)
    env = AtariARIWrapper(env)
    stop_collecting = False
    frame_count = 0
    while not stop_collecting:
        # if len(frames) % 5:
        #     print("Episode %i"%(len(frames)))
        done = False
        env.reset()
        ep_frames = []
        ep_labels = appendabledict()
        ep_actions = []
        while not done:
            action = rng.randint(env.action_space.n)
            obs, reward, done, info = env.step(action)
            label = info["labels"]
            ep_frames.append(torch.tensor(obs))
            ep_labels.append_update(label)
            ep_actions.append(action)
            frame_count += 1
            if max_frames and frame_count == max_frames:
                stop_collecting = True
        if len(ep_frames) > min_episode_length:
            frames.append(torch.stack(ep_frames))
            labels.append(ep_labels)
            actions.append(torch.tensor(ep_actions))
        if max_episodes and len(frames) == max_episodes:
            stop_collecting = True

    if not keep_as_episodes:
        frames = torch.cat(frames)
        labels = flatten_labels(labels)
        actions = torch.cat(actions)

    env.close()


    return frames, actions, labels




class EpisodeDataset(torch.utils.data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, episodes, actions):
        self.episodes = episodes
        self.actions = actions
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
        action = self.actions[ep][step]
        next_obs = self.episodes[ep][step + 1] / 255.

        return obs, action, next_obs







