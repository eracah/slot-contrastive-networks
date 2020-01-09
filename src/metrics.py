from copy import deepcopy
from scipy.optimize import linear_sum_assignment as lsa
from src.utils import prepend_prefix, append_suffix
import wandb
import pandas as pd
import numpy as np
from scipy.stats import entropy
from src.evaluate import LinearProbeTrainer, GBTProbeTrainer,\
    MLPProbeTrainer, postprocess_raw_metrics, AttentionProbeTrainer

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



    f_tr_val = np.concatenate((f_tr, f_val))
    y_tr.extend_update(y_val)
    probe_dict = {"mlp": MLPProbeTrainer, "gbt": GBTProbeTrainer, "linear": LinearProbeTrainer}
    for probe_name, probe_trainer in probe_dict.items():
        # pandas df slot x factors
        slotwise_expl_df = compute_slotwise_explicitness(f_tr_val, y_tr, f_test, y_test, args, probe_trainer=probe_trainer)

        # dict key: factor name, value: explicitness of matched slot with that factor
        assigned_slot_expl = compute_matched_slot_explicitness(slotwise_expl_df)
        # dict key: factor name, value: explicitness for that factor from concatenated slots
        cat_slots_expl = compute_concatenated_slots_explicitness(f_tr_val, y_tr, f_test, y_test, args,
                                                                 probe_trainer=probe_trainer)

        slotwise_expl_dict = convert_slotwise_df_to_flat_dict(slotwise_expl_df)
        log_metrics(slotwise_expl_dict, prefix="slotwise_explicitness_" + probe_name + "_", suffix="_f1")
        postprocess_and_log_metrics(assigned_slot_expl, prefix="assigned_slot_explicitness_" + probe_name + "_",
                                    suffix="_f1")
        postprocess_and_log_metrics(cat_slots_expl, prefix="concatenated_slot_explicitness_" + probe_name + "_",
                                    suffix="_f1")



# compute slot-wise
def compute_slotwise_explicitness(f_tr, y_tr, f_test, y_test, args, probe_trainer=LinearProbeTrainer):
    f1s = []
    num_slots = f_tr.shape[1]
    for i in range(num_slots):
        trainer = probe_trainer(epochs=args.epochs,
                                      lr=args.probe_lr,
                                      patience=args.patience)
        sl_tr, sl_test = f_tr[:, i], f_test[:, i]

        test_f1score, weights = trainer.train_test(sl_tr, y_tr, sl_test, y_test)

        f1s.append(deepcopy(test_f1score))

    return pd.DataFrame(f1s)


def compute_matched_slot_explicitness(slotwise_explicitness):
    f1_np = slotwise_explicitness.to_numpy()
    row_ind, col_ind = lsa(-f1_np)
    inds = list(zip(row_ind, col_ind))
    assigned_slot_f1s = {slotwise_explicitness.columns[factor_num]: f1_np[slot_num, factor_num] for
                         (slot_num, factor_num) in inds}

    return assigned_slot_f1s

def compute_concatenated_slots_explicitness(f_tr, y_tr, f_test, y_test, args, probe_trainer=LinearProbeTrainer):

    f_tr_cat = f_tr.reshape(f_tr.shape[0],-1)
    f_test_cat = f_test.reshape(f_test.shape[0], -1)

    trainer = probe_trainer(epochs=args.epochs,
                                  lr=args.probe_lr,
                                  patience=args.patience)

    cat_test_f1, weights = trainer.train_test(f_tr_cat, y_tr, f_test_cat, y_test)
    return cat_test_f1


# compute disentangling
def compute_sap(df, probe_name="linear"):
    saps_compactness = append_suffix(compute_SAP(df), "_f1_sap_compactness_" + probe_name + '_probe')

    avg_sap_compactness = np.mean(list(saps_compactness.values()))
    f1_maxes = dict(df.max())
    f1_maxes = postprocess_raw_metrics(f1_maxes)
    f1_maxes = {k:v for k, v in f1_maxes.items() if "avg" in k}
    f1_maxes = prepend_prefix(f1_maxes, "best_slot_for_each_f1_" + probe_name + '_probe_')

def compute_SAP(df):
        return {str(k): np.abs(df.nlargest(2, [k])[k].diff().iloc[1]) for k in df.columns}


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



# def compute_slot_importances_from_feat_importances(feat_imps, num_slots, slot_len):
#     importances = []
#     for k, feat_imp in var_names:
#         raw_imp = np.abs(feat_imps[k])
#         normalized_imp = raw_imp / raw_imp.sum(axis=0)
#         class_avged_imp = normalized_imp.mean(axis=1)
#         importances.append(class_avged_imp)
#     imp = np.stack(importances)
#     # split into importances for the weights for each slot and sum to get importances for each slot
#     slot_importances = imp.reshape(len(var_names), num_slots, slot_len).sum(axis=2)