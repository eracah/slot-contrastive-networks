"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from gym.wrappers import TimeLimit
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from src.data.data_collection import get_transitions,  EpisodeDataset
from src.data.stdim_dataloader import get_stdim_eval_dataloader
import argparse
import torch


from atariari.benchmark.wrapper import AtariARIWrapper
# noinspection PyUnresolvedReferences

import gym
from gym import logger


from PIL import Image

import numpy as np


from torch.utils import data


import matplotlib.pyplot as plt
class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


def get_cswm_data(env_name, seed, num_episodes=1000):
    logger.set_level(logger.INFO)

    env = gym.make(env_name)

    np.random.seed(seed)
    env.action_space.seed(seed)
    env.seed(seed)

    agent = RandomAgent(env.action_space)

    episode_count = num_episodes
    reward = 0
    done = False

    crop = None
    warmstart = None
    if env_name == 'PongDeterministic-v4':
        crop = (35, 190)
        warmstart = 58
    elif env_name == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warmstart = 50
    else:
        crop = (35, 190)
        warmstart = 58

    max_episode_steps = warmstart + 11
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = AtariARIWrapper(env)
    replay_buffer = []

    for i in range(episode_count):

        replay_buffer.append({
            'obs': [],
            'action': [],
            'next_obs': [],
            'label': []
        })

        ob = env.reset()


        # Burn-in steps
        for _ in range(warmstart):
            action = agent.act(ob, reward, done)
            ob, _, _, _ = env.step(action)
        prev_ob = crop_normalize(ob, crop)
        ob, _, _, info = env.step(0)
        ob = crop_normalize(ob, crop)

        while True:
            replay_buffer[i]['obs'].append(
                np.concatenate((ob, prev_ob), axis=0))
            prev_ob = ob
            replay_buffer[i]["label"].append(info["labels"])
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            ob = crop_normalize(ob, crop)

            replay_buffer[i]['action'].append(action)
            replay_buffer[i]['next_obs'].append(
                np.concatenate((ob, prev_ob), axis=0))


            if done:
                break


        if i % 10 == 0:
            print("iter "+str(i))

    return replay_buffer


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, buffer, eval=False):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = buffer
        self.eval = eval

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = torch.tensor(self.experience_buffer[ep]['obs'][step], dtype=torch.float32)
        action = torch.tensor(self.experience_buffer[ep]['action'][step], dtype=torch.int64)
        next_obs = torch.tensor(self.experience_buffer[ep]['next_obs'][step], dtype=torch.float32)
        label = self.experience_buffer[ep]['label'][step]
        if self.eval:
            return [obs, torch.tensor(list(label.values()))]
        else:
            return [obs, action, next_obs]


def get_cswm_dataloader(args, mode="train"):
    if mode == "train":
        return get_cswm_train_dataloader(args)
    else:
        return get_cswm_eval_dataloader(args)

def get_cswm_train_dataloader(args):
    num_tr_episodes = round(0.8 * args.num_episodes)

    eps, actions, _ = get_transitions(args, seed=args.seed, max_episodes=args.num_episodes)
    tr_eps, val_eps = eps[:num_tr_episodes], eps[num_tr_episodes:]
    tr_actions, val_actions = actions[:num_tr_episodes], actions[num_tr_episodes:]


    tr_dataset = EpisodeDataset(tr_eps, tr_actions)
    val_dataset = EpisodeDataset(val_eps, val_actions)

    tr_dl = data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return tr_dl, val_dl


def get_cswm_eval_dataloader(args):
    return get_stdim_eval_dataloader(args)






