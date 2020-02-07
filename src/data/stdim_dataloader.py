from torch.utils.data import DataLoader, TensorDataset
try:
    import wandb
except:
    pass
import torch
from src.data.data_collection import get_transitions, get_probe_data, EpisodeDataset

def get_stdim_dataloader(args, mode="train"):
    if mode == "train":
        return get_stdim_train_dataloader(args)
    else:
        return get_stdim_eval_dataloader(args)


def get_stdim_train_dataloader(args):
    eps, actions, _ = get_transitions(args, max_frames=args.num_frames)
    num_episodes = len(eps)
    num_tr_episodes = round(0.8 * num_episodes)
    tr_eps, val_eps = eps[:num_tr_episodes], eps[num_tr_episodes:]
    tr_actions, val_actions = actions[:num_tr_episodes], actions[num_tr_episodes:]
    tr_dataset = EpisodeDataset(tr_eps, tr_actions)
    val_dataset = EpisodeDataset(val_eps, val_actions)

    tr_dl = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return tr_dl, val_dl

def get_stdim_eval_dataloader(args):
    data = get_probe_data(args)
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = data
    tr_dl = make_labelled_dataloader(tr_eps,tr_labels,args.batch_size)
    val_dl = make_labelled_dataloader(val_eps, val_labels, args.batch_size)
    test_dl = make_labelled_dataloader(test_eps, test_labels, args.batch_size)
    label_keys = list(tr_labels.keys())

    return tr_dl, val_dl, test_dl, label_keys

def make_labelled_dataloader(eps, label_dict, batch_size):

    labels = torch.tensor(list(label_dict.values())).long()
    labels_tensor = labels.transpose(1, 0)
    ds = TensorDataset(eps, labels_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl






