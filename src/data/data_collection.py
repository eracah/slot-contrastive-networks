from typing import List, Any

from torch.utils.data import DataLoader
from src.data.wrappers import wrap_atari_env
from atariari.benchmark.wrapper import AtariARIWrapper
import gym
from gym.wrappers import FrameStack, TimeLimit
try:
    import wandb
except:
    pass
from src.utils import appendabledict

from itertools import chain
import torch
from scipy.stats import entropy as compute_entropy
from collections import Counter
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


def get_transitions(args, seed=42, keep_as_episodes=True, min_episode_length=8, max_frames=None, max_episodes=None):
    """workhorse function: collect frames, actions, and labels and split into episodes"""
    frames, labels, actions = [], [], []
    env = gym.make(args.env_name)
    env.seed(seed)
    env = wrap_atari_env(env, args)
    env = AtariARIWrapper(env)
    stop_collecting = False
    frame_count = 0
    while not stop_collecting:
        print("Episode %i"%(len(frames)))
        done = False
        env.reset()
        ep_frames = []
        ep_labels = appendabledict()
        ep_actions = []
        while not done:
            action = env.action_space.sample()
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

    env.close()


    return frames, actions, labels



def get_probe_data(args):
    """get frames and labels for probing (no need to split by episode"""
    num_tr_frames = round(0.7 * args.num_frames)
    num_val_frames = round(0.1 * args.num_frames)
    frames, _, labels = get_transitions(args, seed=args.seed, keep_as_episodes=False, max_frames=args.num_frames)
    num_frames = frames.shape[0]
    tr_frames, val_frames, test_frames = frames[:num_tr_frames], frames[num_tr_frames:num_tr_frames + num_val_frames], frames[num_tr_frames + num_val_frames:]
    tr_labels, val_labels, test_labels = appendabledict(**labels.subslice(slice(0, num_tr_frames))),\
                                         appendabledict(**labels.subslice(slice(num_tr_frames, num_tr_frames + num_val_frames))),\
                                         appendabledict(**labels.subslice(slice(num_tr_frames + num_val_frames, num_frames)))


    tr_labels, val_labels, test_labels = remove_low_entropy_labels([tr_labels, val_labels, test_labels], entropy_threshold=args.entropy_threshold)
    test_frames, test_labels = remove_duplicates(test_frames, test_labels, tr_frames, val_frames)

    return tr_frames, val_frames, tr_labels, val_labels, test_frames, test_labels

def flatten_labels(eps_labels):
    labels = appendabledict()
    for ep_label in eps_labels:
        labels.extend_update(ep_label)
    return labels

def remove_low_entropy_labels(episode_labels, entropy_threshold=0.6):
    """remove any state variable, whose distribution of realizations has low entropy"""
    low_entropy_labels = []
    all_labels = flatten_labels(episode_labels)
    for k,v in all_labels.items():
        vcount = np.asarray(list(Counter(v).values()))
        v_entropy = compute_entropy(vcount)
        if v_entropy < entropy_threshold:
            print("Deleting {} for being too low in entropy! Sorry, dood!".format(k))
            low_entropy_labels.append(k)

    for ep_label in episode_labels:
        for k in low_entropy_labels:
            ep_label.pop(k)
    return episode_labels

def remove_duplicates(test_eps, test_labels, *ref_eps):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    num_test_frames = test_eps.shape[0]
    ref_set = []
    for eps in ref_eps:
        ref_set.extend([x.numpy().tostring() for x in eps])
    ref_set = set(ref_set)

    filtered_test_inds = [i for i, obs in enumerate(test_eps) if obs.numpy().tostring() not in ref_set]
    test_eps = torch.stack([test_eps[i] for i in filtered_test_inds])
    filtered_test_labels = appendabledict()
    filtered_test_labels.extend_updates([test_labels.subslice(slice(i, i + 1)) for i in filtered_test_inds])
    test_labels = filtered_test_labels

    dups = num_test_frames - test_eps.shape[0]
    print('Duplicates: {}, New Test Len: {}'.format(dups, test_eps.shape[0]))
    return test_eps, test_labels


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







