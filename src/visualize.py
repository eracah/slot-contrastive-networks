from torch import nn
from torch.utils.data import RandomSampler, BatchSampler
import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def generate_frames_for_viz(episodes, num_frames=8):
    indices = np.random.randint(len(episodes), size=(num_frames,))
    episodes_batch = [episodes[i] for i in indices]
    xs = []
    for ep_ind, episode in enumerate(episodes_batch):
        # Get one sample from this episode
        t = np.random.randint(len(episode))
        xs.append(episode[t])

    xs = torch.stack(xs) / 255.
    return xs

def get_grid_of_frames(frames, num_repeat):
    num_frames = frames.shape[0]
    x_repeat = frames.repeat(1, num_repeat, 1, 1).numpy().reshape(num_repeat*num_frames,*frames.shape[1:])
    xgrid = make_grid(torch.tensor(x_repeat), nrow=num_repeat, padding=0).detach().numpy()
    return xgrid[0]


def get_grid_of_fmaps(encoder, frames, num_repeat):
    fmaps, slot_fmaps = encoder.get_fmaps(frames)
    slot_fmaps = slot_fmaps.detach()
    fm_upsample = nn.functional.interpolate(slot_fmaps, size=frames.shape[-2:], mode="bilinear")
    fms = fm_upsample.shape
    fmu = fm_upsample.reshape(fms[0] * fms[1], 1, *fms[2:])
    fgrid = make_grid(fmu, nrow=num_repeat, padding=0).detach().numpy()
    return fgrid[0]


def superimpose_mask(xgrid, mask_grid):
    fig, ax = plt.subplots()
    im1 = ax.imshow(xgrid, cmap=plt.cm.jet)
    im2 = ax.imshow(mask_grid, cmap=plt.cm.jet, alpha=0.7)
    ax.axis("off")
    return fig

def plot_fmaps(encoder, episodes, num_repeat):
    frames = generate_frames_for_viz(episodes, num_frames=8)
    xgrid = get_grid_of_frames(frames, num_repeat)
    fm_grid = get_grid_of_fmaps(encoder, frames, num_repeat)
    fig = superimpose_mask(xgrid, fm_grid)
    return fig
