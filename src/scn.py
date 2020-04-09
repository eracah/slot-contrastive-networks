import torch
import torch.nn as nn
from src.utils import calculate_accuracy

class SCNModel(nn.Module):
    def __init__(self, args, encoder, device=torch.device('cpu'), wandb=None, ablations=[]):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.wandb = wandb
        self.num_slots = self.args.num_slots
        self.slot_len = self.args.slot_len
        self.ablations = ablations
        self.score_matrix_1 = nn.Linear(self.slot_len, self.slot_len)
        self.score_matrix_2 = nn.Linear(self.slot_len, self.slot_len)
        self.device = device

    def calc_loss1(self, slots_t, slots_pos):
        """Loss 1: Does a pair of slot vectors from the same slot
                   come from consecutive (or within a small window) time steps or not?"""
        batch_size, num_slots, slot_len = slots_t.shape

        # logits: num_slots x batch_size x batch_size
        #        for each slot, for each example in the batch, dot prodcut with every other example in batch
        logits = torch.matmul(self.score_matrix_1(slots_t).transpose(1, 0),
                              slots_pos.permute(1, 2, 0))

        inp = logits.reshape(num_slots*batch_size, -1)
        target = torch.cat([torch.arange(batch_size) for i in range(num_slots)]).to(self.device)
        loss1 = nn.CrossEntropyLoss()(inp, target)
        acc1 = calculate_accuracy(inp.detach().cpu().numpy(), target.detach().cpu().numpy())

        if self.training:
            self.wandb.log({"tr_acc1": acc1})
            self.wandb.log({"tr_loss1": loss1.item()})
        else:
            self.wandb.log({"val_acc1": acc1})
            self.wandb.log({"val_loss1": loss1.item()})
        return loss1


    def calc_loss2(self, slots_t, slots_pos):
        """Loss 2:  Does a pair of vectors close in time come from the
                          same slot of different slots"""


        batch_size, num_slots, slot_len = slots_t.shape
        # logits: batch_size x num_slots x num_slots
        #        for each example (set of 8 slots), for each slot, dot product with every other slot at next time step
        logits = torch.matmul(self.score_matrix_2(slots_t),
                              slots_pos.transpose(2,1))
        inp = logits.reshape(batch_size * num_slots, -1)
        target = torch.cat([torch.arange(num_slots) for i in range(batch_size)]).to(self.device)
        loss2 = nn.CrossEntropyLoss()(inp, target)
        acc2  = calculate_accuracy(inp.detach().cpu().numpy(), target.detach().cpu().numpy())
        if self.training:
            self.wandb.log({"tr_acc2": acc2})
            self.wandb.log({"tr_loss2": loss2.item()})
        else:
            self.wandb.log({"val_acc2": acc2})
            self.wandb.log({"val_loss2": loss2.item()})
        return loss2


    def calc_loss(self, xt, a, xtp1):
        slots_t, slots_pos = self.encoder(xt), self.encoder(xtp1)
        loss1 = self.calc_loss1(slots_t, slots_pos)
        if "loss1-only" in self.ablations:
            loss = loss1
        else:
            loss2 = self.calc_loss2(slots_t, slots_pos)
            loss = loss1 + loss2
        return loss

