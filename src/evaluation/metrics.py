import numpy as np
import pandas as pd
import copy
from scipy.stats import entropy


from src.utils import all_localization_keys, rename_state_var_to_obj_name

def select_just_localization_rows(array, label_keys):
    ind = pd.Index(label_keys, name="sv_name")
    df = pd.DataFrame(array, index=ind)
    loc_keys = [k for k in label_keys if k in all_localization_keys]
    localization_array = df.loc[loc_keys].to_numpy()
    return localization_array, loc_keys

def average_over_obj(keys, data):
    ind = pd.Index(keys, name="sv_name")
    df = pd.DataFrame(data, index=ind)
    df = df.rename(index=rename_state_var_to_obj_name)
    df = df.groupby("sv_name").mean()
    return df.to_numpy()







def calc_slot_importances_from_weights(weights, num_slots):
    """Computes the feature importances aka slot importances given
    weights from a linear regressor or mlp regressor

    Arguments:
        weights (np.ndarray): array of size -> (num_labels, num_slots*slot_len)
                              the weights of the linear regressor
        num_slots (int): the number of slots in the model

    Returns:
        slot_importances (np.ndarray): array of size -> (num_labels, num_slots)
                                        how important each slot was in regressing each label
                                        np.sum(slot_importances, axis=2) should equal 1

    """

    num_state_var, g_vec_len = weights.shape
    slot_weights = weights.reshape((num_state_var, num_slots, -1))
    slot_magnitudes = np.abs(slot_weights)

    #sum all the weights (there should be "slot_len" of them that correspond to a slot
    raw_slot_importances = np.sum(slot_magnitudes, axis=2)

    # normalize slot importances
    slot_importances = raw_slot_importances / np.sum(raw_slot_importances,
                                                     axis=1,
                                                     keepdims=True)
    return slot_importances



def compute_dci_c(importances):
    """Compute DCI Completeness
     (Eastwood, C. and Williams, C.K., 2018. A framework for the
     quantitative evaluation of disentangled representations)

    measures compactness!
    aka on average how many slots encode a given state variables
    the fewer slots encode the variable the higher the dci_c score is

     Arguments:
         importances (np.ndarray): a size (num_state_variables, num_slots) array
                                            which sums to 1 across dimension 1

    Return:
        dci_c (int): average completeness score across all state variables

     """

    # each entropy computed is the entropy for a state variable over all the slots
    state_var_entropy = entropy(importances.T, base=importances.shape[1])
    dci_cs = 1 - state_var_entropy
    dci_c = dci_cs.mean()

    return dci_c

def compute_dci_d(importances):
    """Compute DCI Disentangling
     (Eastwood, C. and Williams, C.K., 2018. A framework for the
     quantitative evaluation of disentangled representations)

    measures modularity
    aka on average how many state variables does a slot encode
    the fewer state variables that a slot encodes the the higher the dci_d score is
     Arguments:
         importances (np.ndarray): a size (num_state_variables, num_slots) array
                                            which sums to 1 across dimension 1

    Return:
        dci_d (int): average completeness score across all state variables

     """

    # normalize across the state_variable dimension, so we get the importance of
    # each state variable for a slot
    slot_normalized_importances = importances / importances.sum(axis=0)
    sni = slot_normalized_importances

    # each entropy computed is the entropy for a slot over all the state variables
    slot_entropy = entropy(sni, base=importances.shape[0])
    dci_ds = 1 - slot_entropy
    dci_d = dci_ds.mean()

    return dci_d
