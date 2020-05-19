from torch.utils.data import DataLoader, TensorDataset
try:
    import wandb
except:
    pass
import torch
from src.data.data_collection import get_transitions, EpisodeDataset
from src.utils import appendabledict, remove_duplicates, remove_low_entropy_labels

def get_dataloaders(args, keep_as_episodes=True, test_set=False, label_keys=False):
    data, actions, labels = get_transitions(args,
                                           max_frames=args.num_frames,
                                           keep_as_episodes=keep_as_episodes)

    if not keep_as_episodes:
        labels = remove_low_entropy_labels(labels, entropy_threshold=args.entropy_threshold)
    all_data, all_actions, all_labels = preprocess_data(data, actions, labels,
                                                        args, keep_as_episodes, test_set)

    dataloaders = []
    for data, action, label in zip(all_data, all_actions, all_labels):
        dataloader = create_dataloader(data, action, label, args.batch_size, keep_as_episodes)
        dataloaders.append(dataloader)

    if label_keys:
        if not keep_as_episodes:
            lab_keys = list(labels.keys())
        else:
            lab_keys = list(labels[0].keys())
        return dataloaders, lab_keys
    else:
        return dataloaders

def create_dataloader(data, action, label, batch_size, keep_as_episodes=True):
    labels = torch.tensor(list(label.values())).long()
    labels_tensor = labels.transpose(1, 0)
    dataset = EpisodeDataset(data, action) if keep_as_episodes else TensorDataset(data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def preprocess_data(data, actions, labels, args, keep_as_episodes=True, test_set=False):
    num_datapoints = len(data)
    slices = get_slices(num_datapoints, test_set=test_set)
    all_data = split_data(data, *slices)
    all_actions = split_data(actions, *slices)
    if not keep_as_episodes:
        all_labels = split_labels(labels, *slices)
        if test_set:
            all_data, all_labels = remove_duplicates(all_data, all_labels)
    else:
        all_labels = labels
    return all_data, all_actions, all_labels



def split_data(data, *slices):
    all_data = []
    for slice_ in slices:
        all_data.append(data[slice_])
    return all_data

def split_labels(labels,*slices):
    all_labels = []
    for slice_ in slices:
        sliced_label = labels.subslice(slice_)
        label = appendabledict(**sliced_label)
        all_labels.append(label)
    return all_labels

def get_slices(total_datapoints, test_set = False):
    if test_set:
        num_tr = round(0.7 * total_datapoints)
        num_val = round(0.1 * total_datapoints)
        tr_slice, val_slice, test_slice = slice(0, num_tr), \
                                          slice(num_tr, num_tr + num_val), \
                                          slice(num_tr + num_val, total_datapoints)
        slices = [tr_slice, val_slice, test_slice]
    else:
        num_tr = round(0.8 * total_datapoints)
        tr_slice, val_slice = slice(0, num_tr), slice(num_tr, total_datapoints)
        slices = [tr_slice, val_slice]
    return slices





