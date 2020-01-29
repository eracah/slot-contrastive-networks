import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.utils import EarlyStopping
from src.encoders import SlotEncoder
from src.utils import calculate_accuracy
from src.data.data import EpisodeDataset

import time

class BilinearScoreFunction(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class SCNTrainer(object):
    def __init__(self, args, device=torch.device('cpu'), wandb=None, ablation="none"):
        self.encoder = SlotEncoder(args.num_channels, args.slot_len, args.num_slots, args).to(device)
        self.args = args
        self.wandb = wandb
        self.patience = self.args.patience
        self.score_matrix_1 = nn.Linear(self.encoder.slot_len, self.encoder.slot_len).to(device)
        self.score_matrix_2 = nn.Linear(self.encoder.slot_len, self.encoder.slot_len).to(device)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = device
        self.ablation = ablation
        self.optimizer = torch.optim.Adam(list(self.score_matrix_1.parameters()) + list(self.score_matrix_2.parameters()) +\
                                          list(self.encoder.parameters()),
                                          lr=args.lr, eps=1e-5)
        self.early_stopper1 = EarlyStopping(patience=self.patience, verbose=False, name="infonce_loss1")
        self.early_stopper2 = EarlyStopping(patience=self.patience, verbose=False, name="infonce_loss2")




    def compute_loss1(self, slots_t, slots_pos ):
        """Loss 1: Does a pair of slot vectors from the same slot
                   come from consecutive (or within a small window) time steps or not?"""
        batch_size = slots_t.shape[0]
        loss1is,  acc1is = [], []
        for i in range(self.encoder.num_slots):
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
        return loss1, acc1is


    def compute_loss2(self, slots_t, slots_pos):
        """Loss 2:  Does a pair of vectors close in time come from the
                          same slot of different slots"""
        batch_size = slots_t.shape[0]
        loss2is, acc2is = [], []
        for i in range(self.encoder.num_slots):
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
            mask = torch.arange(self.encoder.num_slots) != i
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

        return loss2, acc2is

    def do_one_epoch(self, epoch, dataloader):
        mode = "train" if self.encoder.training and self.score_matrix_1.training and self.score_matrix_2.training  else "val"
        losses, loss1s, loss2s = [], [], []
        accs, acc1s, acc2s = [], [], []
        iterations = 0
        t0 = time.time()
        for x, x_pos in dataloader:
            slots_t, slots_pos = self.encoder(x), self.encoder(x_pos)

            loss1, acc1_correct_or_nots = self.compute_loss1(slots_t, slots_pos)
            loss2, acc2_correct_or_nots = self.compute_loss2(slots_t, slots_pos)
            loss = loss1 + loss2

            self.optimizer.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            loss1s.append(loss1.detach().cpu().numpy())
            loss2s.append(loss2.detach().cpu().numpy())
            acc1s.extend(acc1_correct_or_nots)
            acc2s.extend(acc2_correct_or_nots)
            iterations += 1



        epoch_loss = np.mean(losses)
        epoch_loss1 = np.mean(loss1s)
        epoch_loss2 = np.mean(loss2s)

        epoch_acc1 = np.mean(acc1s)
        epoch_acc2 = np.mean(acc2s)
        epoch_acc = np.mean(acc1s + acc2s)


        loss_terms = {mode + "_loss1":epoch_loss1, mode + "_loss2":epoch_loss2}
        acc_terms = {mode + "_acc1": epoch_acc1, mode + "_acc2": epoch_acc2}
        self.log_results(epoch_loss, epoch_loss1, epoch_loss2, epoch_acc, epoch_acc1, epoch_acc2, mode=mode)
        print("\t\tnum iterations: {}".format(iterations))
        print("\t\tepoch time: {}".format(time.time() - t0))
        if mode == "val":
            self.early_stopper1(-epoch_loss1, self.encoder)
            self.early_stopper2(-epoch_loss2, self.encoder)

        return epoch_loss, epoch_acc, acc_terms, loss_terms

    def train(self, tr_eps, val_eps):
        tr_dataset, val_dataset = EpisodeDataset(tr_eps), EpisodeDataset(val_eps)
        tr_dl, val_dl = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True),  DataLoader(val_dataset, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch))
            self.encoder.train(), self.score_matrix_1.train(), self.score_matrix_2.train()
            tr_loss,  tr_acc,  tr_acc_terms, tr_loss_terms = self.do_one_epoch(epoch, tr_dl)
            self.wandb.log(tr_loss_terms, step=epoch)
            self.wandb.log(tr_acc_terms, step=epoch)

            self.encoder.eval(), self.score_matrix_1.eval(), self.score_matrix_2.eval()
            val_loss, val_acc, val_acc_terms, val_loss_terms = self.do_one_epoch(epoch, val_dl)
            self.wandb.log(val_loss_terms, step=epoch)
            self.wandb.log(val_acc_terms, step=epoch)
            self.wandb.log(dict(tr_loss=tr_loss, val_loss=val_loss, tr_acc=tr_acc, val_acc=val_acc), step=epoch)
            if self.early_stopper1.early_stop and self.early_stopper2.early_stop:
                break
        return self.encoder

    def log_results(self, loss, loss1, loss2, acc, acc1, acc2, mode=""):
        print("\t{}: ".format(mode), flush=True)
        print("\t\tLoss: {0:.4f}".format(loss), flush=True)
        print("\t\t\tLoss1: {0:.4f}".format(loss1), flush=True)
        print("\t\t\tLoss2: {0:.4f}".format(loss2), flush=True)
        print("\t\tAcc: {0:.2f}%".format(100*acc), flush=True)
        print("\t\t\tAcc1: {0:.2f}%".format(100 * acc1), flush=True)
        print("\t\t\tAcc2: {0:.2f}%".format(100 * acc2), flush=True)
