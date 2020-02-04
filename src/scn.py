import torch
import torch.nn as nn
import numpy as np
from src.utils import EarlyStopping
from src.utils import calculate_accuracy

import time

class BilinearScoreFunction(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class SCNModel(nn.Module):
    def __init__(self, args, encoder, device=torch.device('cpu'), wandb=None, ablation="none"):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.wandb = wandb
        self.num_slots = self.args.num_slots
        self.embedding_dim = self.args.embedding_dim
        self.ablation = ablation
        self.score_matrix_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.ablation != "hybrid":
            self.score_matrix_2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.device = device

    def calc_loss1(self, slots_t, slots_pos):
        """Loss 1: Does a pair of slot vectors from the same slot
                   come from consecutive (or within a small window) time steps or not?"""
        batch_size = slots_t.shape[0]
        loss1is,  acc1is = [], []
        for i in range(self.num_slots):
            # batch_size, slot_len
            slot_t_i, slot_pos_i = slots_t[:, i], slots_pos[:, i]
            slot_t_i_transformed = self.score_matrix_1(slot_t_i)  # transforms each slot into some interemdiate representation the same len as the slots]
            # dot product of each slot at time t in batch with every slot at time t+1 in the batch
            # result is batch_size x batch_size matrix (for every slot_i in batch for every slot_pos_i vector batch what is dot product)
            # each element of the diagonal of the matrix the dot product of a slot vector with the same slot vector in the same episode at one time step in the future
            # so each element of the diagonal is a positive logit and off-diagonals are negative logits
            logits = torch.matmul(slot_t_i_transformed, slot_pos_i.T)
            # thus we do multi-class classifcation where the label for each row is equal to the index of the row
            # ground truth for row 0 is 0, row 1 is 1, etc.
            ground_truth = torch.arange(batch_size).to(self.device)

            loss1i = nn.CrossEntropyLoss()(logits, ground_truth)

            _, correct_or_not = calculate_accuracy(logits.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
            loss1is.append(loss1i)
            acc1is.extend(correct_or_not)
        loss1 = torch.mean(torch.stack(loss1is))
        return loss1#, acc1is


    def calc_loss2(self, slots_t, slots_pos):
        """Loss 2:  Does a pair of vectors close in time come from the
                          same slot of different slots"""
        batch_size = slots_t.shape[0]
        loss2is, acc2is = [], []
        for i in range(self.num_slots):
            slot_t_i = slots_t[:, i]
            slot_t_i_transformed = self.score_matrix_2(slot_t_i)  # transforms each slot into some interemdiate representation the same len as the slots
            # this results in a batch_size x batch_size x num_slots matrix
            # which is the dot product of every ith_slot vector at time t in the batch with every set of slots at time t+1 in the batch
            # so scores[j,k,l] is the jth slot_t_i vector in the batch multiplied by the lth slot in the kth set of slots in the batch of t+1 slots
            scores = torch.matmul(slots_pos, slot_t_i_transformed.T).transpose(2, 1)
            # each element of the diagonal of this matrix is the dot product of a slot vector with the same slot vector in the same episode at one time step in the future
            # we only want slot_i's from same episode to be paired
            pos_logits = scores[:, :, i].diag()
            # mask out the ith slot for the neg_logits (we only want mismatched slots to be paired together for neg logits)
            mask = torch.arange(self.num_slots) != i
            neg_logits = scores[:, :, mask]
            neg_logits = neg_logits.reshape(batch_size, -1)
            # now we concatenate pos_logits column with negative logits matrix to create
            # bactch_size x set of logits matrix
            # the shape will be batch_size x 1 + batch_size * (num_slots - 1)
            logits = torch.cat((pos_logits[:, None], neg_logits), dim=1)
            # ground truth index is always 0 for every logit vector in batch
            # because we put the column of positive logits on the left
            ground_truth = torch.zeros(batch_size).long().to(self.device)
            loss2i = nn.CrossEntropyLoss()(logits, ground_truth)
            _, acc2i = calculate_accuracy(logits.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
            acc2is.extend(acc2i)
            loss2is.append(loss2i)

        loss2 = torch.mean(torch.stack(loss2is))

        return loss2 #, acc2is


    def calc_loss(self, xt, a, xtp1):
        slots_t, slots_pos = self.encoder(xt), self.encoder(xtp1)
        loss1 = self.calc_loss1(slots_t, slots_pos)
        loss2 = self.calc_loss2(slots_t, slots_pos)
        loss = loss1 + loss2
        return loss

    def compute_acc1(self):
        pass
    def compute_acc2(self):
        pass

