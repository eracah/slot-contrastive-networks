import copy
import os
import torch
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from collections import defaultdict
import psutil
import wandb
from atariari.benchmark.categorization import summary_key_dict
from scipy.stats import entropy
from scipy.stats import entropy as compute_entropy
from collections import Counter

def reformat_label_keys(label_keys):
    return [reformat_label_str(label_key)
     for label_key in label_keys]

def reformat_label_str(label_key):
    '''Put _x, _y, or _z to end of string'''
    for suffix in ["_x", "_y", "_z"]:
        if suffix in label_key:
            label_key_substrs = label_key.split(suffix)
            label_key = "".join(label_key_substrs) + suffix
    return label_key

all_localization_keys = []
for category_name, category_keys in summary_key_dict.items():
    if "localization" in category_name:
        reformatted_keys = reformat_label_keys(category_keys)
        all_localization_keys.extend(reformatted_keys)

def get_sample_frame(dataloader):
    sample_frame = next(dataloader.__iter__())[0]
    return sample_frame

def get_sample_label(dataloader):
    sample_label = next(dataloader.__iter__())[-1]
    return sample_label



# def get_label_keys(dataloader):
#     sample_label = get_sample_label(dataloader)
#     sample_label_keys = sample_label.keys()
#     label_keys = [reformat_label_str(label_key)
#                   for label_key in sample_label_keys]
#     return label_keys


def flatten_labels(eps_labels):
    labels = appendabledict()
    for ep_label in eps_labels:
        labels.extend_update(ep_label)
    return labels

def remove_low_entropy_labels(labels, entropy_threshold=0.6):
    """remove any state variable, whose distribution of realizations has low entropy"""
    low_entropy_labels = []
    for k,v in labels.items():
        vcount = np.asarray(list(Counter(v).values()))
        v_entropy = compute_entropy(vcount)
        if v_entropy < entropy_threshold:
            print("Deleting {} for being too low in entropy! Sorry, dood!".format(k))
            low_entropy_labels.append(k)
    for k in low_entropy_labels:
        labels.pop(k)

    return labels

def remove_duplicates(frames, labels):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    test_frames = frames[-1]
    ref_frames = frames[:-1]
    test_labels = labels[-1]
    num_test_frames = test_frames.shape[0]
    ref_set = []
    for ref_frame in ref_frames:
        ref_set.extend([x.numpy().tostring() for x in ref_frame])
    ref_set = set(ref_set)

    filtered_test_inds = [i for i, obs in enumerate(test_frames) if obs.numpy().tostring() not in ref_set]
    test_frames = torch.stack([test_frames[i] for i in filtered_test_inds])
    filtered_test_labels = appendabledict()
    filtered_test_labels.extend_updates([test_labels.subslice(slice(i, i + 1)) for i in filtered_test_inds])
    test_labels = filtered_test_labels

    dups = num_test_frames - test_frames.shape[0]
    print('Duplicates: {}, New Test Len: {}'.format(dups, test_frames.shape[0]))
    frames = [*ref_frames, test_frames]
    labels = [*labels[:-1], test_labels]
    return frames, labels

def get_obj_list(label_keys):
    loc_keys = [k for k in label_keys if k in all_localization_keys]
    objs = list(set([rename_state_var_to_obj_name(k) for k in loc_keys]))
    return objs

def rename_state_var_to_obj_name(state_var):
    v = copy.deepcopy(state_var)
    for d in ["_x", "_y", "_z"]:
        v = v.replace(d, "")
    return v

def get_num_objects(label_keys):
    objs = get_obj_list(label_keys)
    num_objs = len(objs)
    return num_objs

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



