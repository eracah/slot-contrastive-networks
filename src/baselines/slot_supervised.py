import torch
import torch.nn as nn
import numpy as np
from src.utils import calculate_accuracy, calculate_multiple_accuracies, \
    get_obj_list, all_localization_keys, rename_state_var_to_obj_name


class SupervisedModel(nn.Module):
    """Trains slot based encoder in a fully supervised way
       Assigning one state variable to each slot and training in a simultaneous multi-task way"""
    def __init__(self, encoder, args, label_keys, wandb=None):
        super().__init__()
        self.wandb = wandb
        self.encoder = encoder
        self.label_keys = label_keys
        self.loc_keys = [k for k in label_keys if k in all_localization_keys]
        self.obj_list = get_obj_list(self.label_keys)



        self.slot_len, self.num_slots = self.encoder.slot_len, self.encoder.num_slots
        self.num_state_variables = len(self.label_keys)
        # reduction = none so we can see every state variable's loss
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # one regressor for each state variable
        self.probe = nn.Linear(self.slot_len, self.num_state_variables)

        self.pred_mask = self.get_pred_mask()
        self.label_mask = self.get_label_mask()


    def get_pred_mask(self):
        """Make mask to mask out which predictions from each slot to use"""
        key_order = []
        mask = torch.zeros(self.num_slots, self.num_state_variables)

        for loc_key in self.loc_keys:
            for obj_ind, obj_key in enumerate(self.obj_list):
                obj_name = rename_state_var_to_obj_name(loc_key)
                if obj_name == obj_key:
                    label_ind = self.label_keys.index(loc_key)
                    mask[obj_ind, label_ind] = 1
                    key_order.append(loc_key)

        assert key_order == self.loc_keys
        return mask.bool()

    def get_label_mask(self):
        label_mask = torch.tensor([1 if k in self.loc_keys else 0 for k in self.label_keys])
        return label_mask.bool()

    def forward(self, x):
        x = x / 255.
        slots = self.encoder(x)
        # apply every regressor/classifier to every slot (later we will index out one unique state variable prediction per slot)
        preds = self.probe(slots)


        # index out the diagonals so each slot has a regression prediction for its own particular state variable
        preds = preds[:, self.pred_mask] #torch.stack([torch.diag(pred) for pred in preds])
        return preds

    def calc_loss(self, x, y):
        y = y[:, self.label_mask].float()
        preds = self.forward(x)
        losses = self.loss_fn(preds, y)

        mode = "tr" if self.training else "val"
        # for logging purposes capture loss per state variable
        sv_losses = losses.mean(axis=0).detach().cpu().numpy()

        #sv_accs = calculate_multiple_accuracies(preds.detach().cpu().numpy().argmax(axis=1), y.detach().cpu().numpy())

        loss_keys = [k + "_" + mode + "_loss" for k in self.loc_keys]
        # acc_keys = [k + "_" + mode + "_acc" for k in self.label_keys]
        # avg_acc = np.mean(sv_accs)
        self.wandb.log(dict(zip(loss_keys, sv_losses)))
        # self.wandb.log(dict(zip(acc_keys, sv_accs)))
        # self.wandb.log({mode + "_acc": avg_acc})

        #main loss is mean over batch and every state variable
        loss = losses.mean()
        return loss
