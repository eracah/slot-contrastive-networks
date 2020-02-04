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

import argparse


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
        ob, _, _, _ = env.step(0)
        ob = crop_normalize(ob, crop)

        while True:
            replay_buffer[i]['obs'].append(
                np.concatenate((ob, prev_ob), axis=0))
            prev_ob = ob

            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            ob = crop_normalize(ob, crop)

            replay_buffer[i]['action'].append(action)
            replay_buffer[i]['next_obs'].append(
                np.concatenate((ob, prev_ob), axis=0))
            replay_buffer[i]["label"].append(info["labels"])

            if done:
                break


        if i % 10 == 0:
            print("iter "+str(i))

    return replay_buffer


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, buffer, labels=False):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = buffer
        self.labels = labels

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

        obs = self.experience_buffer[ep]['obs'][step]
        action = self.experience_buffer[ep]['action'][step]
        next_obs = self.experience_buffer[ep]['next_obs'][step]
        label = self.experience_buffer[ep]['label'][step]
        obs = np.array(obs,dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
        if self.labels:
            return [obs, action, next_obs, np.asarray(list(label.values()))]
        else:
            return [obs, action, next_obs]


def get_cswm_dataloader(args, mode="train"):
    if mode == "train":
        return get_cswm_train_dataloader(args)
    else:
        return get_cswm_eval_dataloader(args)

def get_cswm_train_dataloader(args):

    tr_buffer = get_cswm_data(args.env_name, args.seed, int(0.8*args.num_episodes))
    tr_ds = StateTransitionsDataset(tr_buffer)
    tr_dl = data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_buffer = get_cswm_data(args.env_name, args.seed + 10, int(0.2 * args.num_episodes))
    val_ds = StateTransitionsDataset(val_buffer)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return tr_dl, val_dl


def get_cswm_eval_dataloader(args):
    tr_buffer = get_cswm_data(args.env_name, args.seed, int(0.8*args.num_episodes))
    tr_ds = StateTransitionsDataset(tr_buffer, labels=True)
    tr_dl = data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_buffer = get_cswm_data(args.env_name, args.seed + 10, int(0.1 * args.num_episodes))
    val_ds = StateTransitionsDataset(val_buffer, labels=True)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_buffer = get_cswm_data(args.env_name, args.seed + 20, int(0.1 * args.num_episodes))
    test_ds = StateTransitionsDataset(test_buffer, labels=True)
    test_dl = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return tr_dl, val_dl, test_dl





