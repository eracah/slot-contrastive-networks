from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
from src.utils import prepend_prefix, append_suffix
import wandb
import pandas as pd
import numpy as np
from scipy.stats import entropy
from src.evaluate import  AttentionProbeTrainer
from src.utils import appendabledict, compute_dict_average, append_suffix
from atariari.benchmark.categorization import summary_key_dict

from atariari.benchmark.probe import train_all_probes

def compute_and_log_raw_quant_metrics(args, f_tr, y_tr, f_val, y_val,  f_test, y_test):

    attn_probe = AttentionProbeTrainer(epochs=args.epochs, patience=args.patience, lr=args.probe_lr, type="linear")
    # scores: dict keys: factor name values: probe f1 score for that factor
    # importances_df: pandas df slot x factors
    scores, importances_df = attn_probe.train_test(f_tr, y_tr, f_val, y_val, f_test, y_test)
    imp_dict = convert_slotwise_df_to_flat_dict(importances_df)
    log_metrics(imp_dict, prefix="attn_lin_probe_importances_", suffix="_weights")
    log_metrics(scores, prefix="attn_lin_probe_explicitness_", suffix="_f1")

    attn_probe = AttentionProbeTrainer(epochs=args.epochs, patience=args.patience, lr=args.probe_lr, type="mlp")
    # scores: dict keys: factor name values: probe f1 score for that factor
    # importances_df: pandas df slot x factors
    scores, importances_df = attn_probe.train_test(f_tr, y_tr, f_val, y_val, f_test, y_test)
    imp_dict = convert_slotwise_df_to_flat_dict(importances_df)
    log_metrics(imp_dict, prefix="attn_mlp_probe_importances_", suffix="_weights")
    log_metrics(scores, prefix="attn_mlp_probe_explicitness_", suffix="_f1")


    slotwise_expl_df = get_explicitness_for_every_slot_for_every_factor(f_tr, y_tr, f_val, y_val,  f_test, y_test, args)

    # dict key: factor name, value: explicitness of matched slot with that factor
    matched_slot_expl = compute_matched_slot_explicitness(slotwise_expl_df)
    best_slot_expl = compute_best_slot_explicitness(slotwise_expl_df)
    slotwise_expl_dict = convert_slotwise_df_to_flat_dict(slotwise_expl_df)

    log_metrics(slotwise_expl_dict, prefix="slotwise_explicitness_", suffix="_f1")
    postprocess_and_log_metrics(matched_slot_expl, prefix="assigned_slot_explicitness_",
                                suffix="_f1")
    postprocess_and_log_metrics(best_slot_expl, prefix="best_slot_explicitness_",
                                suffix="_f1")


# compute slot-wise
def get_explicitness_for_every_slot_for_every_factor(f_tr, y_tr, f_val, y_val,  f_test, y_test, args):
    f1s = []
    num_slots = f_tr.shape[1]
    for i in range(num_slots):
        sl_tr, sl_val, sl_test = f_tr[:, i], f_val[:,i], f_test[:, i]
        encoder = None #because inouts are vectors
        representation_len = sl_tr.shape[-1]
        test_acc, test_f1score = train_all_probes(encoder, sl_tr, sl_val,sl_test,y_tr, y_val, y_test, representation_len, args, wandb.run.dir)



        f1s.append(deepcopy(test_f1score))

    return pd.DataFrame(f1s)


def compute_matched_slot_explicitness(slotwise_explicitness_df):
    f1_np = slotwise_explicitness_df.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_f1s = {slotwise_explicitness_df.columns[factor_num]: f1_np[slot_num, factor_num] for
                         (slot_num, factor_num) in inds}

    return assigned_slot_f1s


def compute_best_slot_explicitness(slotwise_explicitness_df):
    return dict(slotwise_explicitness_df.max())


def convert_slotwise_df_to_flat_dict(df):
    dic = {col: df.values[:, i] for i, col in enumerate(df.columns)}
    return dic


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
    category_avgs_dict = compute_category_avgs(metric_dict)
    avg_across_categories = compute_dict_average(category_avgs_dict)
    metric_dict.update(category_avgs_dict)

    metric_dict["overall_avg"] = overall_avg
    metric_dict["across_categories_avg"] = avg_across_categories

    return metric_dict
