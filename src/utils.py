import argparse
import copy
import os
import subprocess

import torch
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from collections import defaultdict
from pathlib import Path
import psutil
import wandb
from atariari.benchmark.categorization import summary_key_dict
from scipy.stats import entropy

all_localization_keys = []
for category_name, category_keys in summary_key_dict.items():
    if "localization" in category_name:
        all_localization_keys.extend(category_keys)


# methods that need encoder trained before
ablations = ["nce", "infonce","shared_score_fxn", "loss1_only", "loss2_only"]
baselines = ["supervised", "random-cnn", "stdim", "cswm"]




def get_channels(args):
    if args.color and args.num_frame_stack == 1:
        num_channels = 3
    elif not args.color:
        num_channels = args.num_frame_stack
    else:
        assert False, "undefined behavior for color and frame stack > 1! "
    return num_channels

def print_memory(name=""):
    process = psutil.Process(os.getpid())
    print("%3.4f GB for %s"%(process.memory_info().rss / 2**30,name), flush=True)  # in bytes


def prepend_prefix(dictionary, prefix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[prefix + k] = v
    return new_dict

def append_suffix(dictionary, suffix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k + suffix] = v
    return new_dict

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def calculate_multiple_f1_scores(preds, labels):
    if len(labels.shape) == 1:
        return calculate_multiclass_f1_score(preds, labels)
    else:
        return [compute_f1_score(preds[:,i], labels[:,i],average="weighted") for i in range(labels.shape[-1])]

def calculate_multiple_accuracies(preds, labels):
    if len(labels.shape) == 1:
        return calculate_multiclass_accuracy(preds, labels)
    else:
        return [calculate_multiclass_accuracy(preds[:,i], labels[:,i]) for i in range(labels.shape[-1])]

def calculate_multiclass_accuracy(preds, labels):
    acc = float(np.sum((preds == labels).astype(int)) / len(labels))
    return acc

def calculate_multiclass_f1_score(preds, labels):
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_f1_score(logits, labels):
    preds = np.argmax(logits, axis=1)
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_accuracy(logits, labels, argmax=True):
    if argmax:
        preds = np.argmax(logits, axis=1)
    else:
        preds = logits
    correct_or_not = (preds == labels).astype(int)
    acc = np.mean(correct_or_not)
    return acc

def log_metrics(dic, prefix, suffix):
    dic = prepend_prefix(dic, prefix)
    dic = append_suffix(dic, suffix)
    wandb.run.summary.update(dic)

def postprocess_and_log_metrics(dic, prefix, suffix):
    dic = postprocess_raw_metrics(dic)
    log_metrics(dic, prefix, suffix)

def compute_dci_d(slot_importances, explicitness_scores, weighted_by_explicitness=True):
    num_factors = len(slot_importances.keys())
    dci_d = 1 - entropy(slot_importances, base=num_factors)
    if weighted_by_explicitness:
        dci_d = explicitness_scores * dci_d
    return dci_d


def compute_category_avgs(metric_dict):
    category_dict = {}
    for category_name, category_keys in summary_key_dict.items():
        category_values = [v for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean


    return category_dict


def postprocess_raw_metrics(metric_dict):
    overall_avg = compute_dict_average(metric_dict)
    overall_localization_avg = np.mean([v for k, v in metric_dict.items() if k in all_localization_keys])
    category_avgs_dict = compute_category_avgs(metric_dict)
    across_categories_localization_avg = np.mean([v for k, v in category_avgs_dict.items() if "localization" in k])
    avg_across_categories = compute_dict_average(category_avgs_dict)
    metric_dict.update(category_avgs_dict)

    metric_dict["overall_avg"] = overall_avg
    metric_dict["across_categories_avg"] = avg_across_categories
    metric_dict["overall_localization_avg"] = overall_localization_avg
    metric_dict["across_categories_localization_avg"] = across_categories_localization_avg


    return metric_dict



def append_suffix(dictionary, suffix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k + suffix] = v
    return new_dict



def compute_dict_average(metric_dict):
    return np.mean(list(metric_dict.values()))

# def save_model(model, envs, save_dir, model_name, use_cuda):
#     save_path = os.path.join(save_dir)
#     try:
#         os.makedirs(save_path)
#     except OSError:
#         pass
#
#     # A really ugly way to save a model to CPU
#     save_model = model
#     if use_cuda:
#         save_model = copy.deepcopy(model).cpu()
#
#     save_model = [save_model,
#                   getattr(get_vec_normalize(envs), 'ob_rms', None)]
#
#     torch.save(save_model, os.path.join(save_path, model_name + ".pt"))




class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    #     def map_(self, func):
    #         for k, v in self.items():
    #             self.__setitem__(k, func(v))

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

         """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        """appends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)

    def append_updates(self, list_of_dicts):
        """appends current dict's values with values from all the other_dicts

        Parameters
        ----------
        list_of_dicts : list[dict]
            A list of dictionaries that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for other_dict in list_of_dicts:
            self.append_update(other_dict)

    def extend_update(self, other_dict):
        """extends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to extend to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for k, v in other_dict.items():
            self.__getitem__(k).extend(v)

    def extend_updates(self, list_of_dicts):
        """extends current dict's values with values from all the other_dicts

        Parameters
        ----------
        list_of_dicts : list[dict]
            A list of dictionaries that you want to extend to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for other_dict in list_of_dicts:
            self.extend_update(other_dict)


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, name=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.
        self.name = name


    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'{self.name} has stopped')

        else:
            self.best_score = score
            self.counter = 0



