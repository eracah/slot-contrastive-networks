import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils import calculate_accuracy

class STDIMModel(nn.Module):
    def __init__(self, encoder, args, embedding_dim, device=torch.device('cpu'), wandb=None):
        super().__init__()
        self.encoder = encoder
        self.fmap_encoder = encoder.fmap_encoder
        self.embedding_dim = embedding_dim
        self.wandb = wandb
        self.args = args

        self.score_fxn1 = nn.Linear(self.embedding_dim, self.fmap_encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.score_fxn2 = nn.Linear(self.fmap_encoder.local_layer_depth, self.fmap_encoder.local_layer_depth).to(device)
        if "structure-loss" in self.args.ablations:
            num_slots = self.args.num_slots
            assert self.embedding_dim % num_slots == 0, "Embedding dim (%i) must be divisible by num_slots (%i)!!!"%(self.embedding_dim, num_slots)
            slot_len = self.embedding_dim // num_slots
            self.score_fxn_3 = nn.Linear(slot_len,slot_len)
        self.device = device

    def calc_loss(self, xt, a, xtp1):
        f_t = self.encoder(xt)
        f_tp1 = self.encoder(xtp1)
        fmap_t = self.fmap_encoder(xt).permute(0, 2, 3, 1)
        fmap_tp1 = self.fmap_encoder(xtp1).permute(0, 2, 3, 1)
        N, sy, sx, d = fmap_tp1.shape

        # Loss 1: Global at time t, f5 patches at time t-1
        glob_score = self.score_fxn1(f_t)
        local_flattened = fmap_tp1.reshape(-1, d)
        # [N*sy*sx, d] @  [d, N] = [N*sy*sx, N ] -> dot product of every global vector in batch with local voxel at all spatial locations for all examples in the batch
        # then reshape to sy*sx, N, N then to sy*sx*N, N
        logits1 = torch.matmul(local_flattened, glob_score.t()).reshape(N, sy * sx, -1).transpose(1, 0).reshape(-1, N)
        # we now have sy*sx N x N matrices where the diagonals correspond to dot product between pairs consecutive in time at the same bagtch index
        # aka the ocrrect answer. So the correct logit index is the diagonal sx*sy times
        target1 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss1 = nn.CrossEntropyLoss()(logits1, target1)


        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        local_t_score = self.score_fxn2(fmap_t.reshape(-1, d))
        transformed_local_t = local_t_score.reshape(N, sy*sx,d).transpose(0,1)
        local_tp1 = fmap_tp1.reshape(N, sy * sx, d).transpose(0, 1)
        logits2 = torch.matmul(transformed_local_t, local_tp1.transpose(1, 2)).reshape(-1, N)
        target2 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss2 = nn.CrossEntropyLoss()(logits2, target2)

        loss = loss1 + loss2
        if "structure-loss" in self.args.ablations:
            loss3 = self.calc_slot_structure_loss(f_t, f_tp1)
            loss += loss3
        acc1 = calculate_accuracy(logits1.detach().cpu().numpy(), target1.detach().cpu().numpy())
        acc2 = calculate_accuracy(logits2.detach().cpu().numpy(), target2.detach().cpu().numpy())
        if self.training:
            self.wandb.log({"tr_acc1": acc1})
            self.wandb.log({"tr_acc2": acc2})
            self.wandb.log({"tr_loss1": loss1.item()})
            self.wandb.log({"tr_loss2": loss2.item()})
        else:
            self.wandb.log({"val_acc1": acc1})
            self.wandb.log({"val_acc2": acc2})
            self.wandb.log({"val_loss1": loss1.item()})
            self.wandb.log({"val_loss2": loss2.item()})
        return loss


    def calc_slot_structure_loss(self, f_t, f_tp1):
        batch_size, global_vec_len = f_t.shape
        num_slots = self.args.num_slots
        try:
            # cut up vector into slots
            slots_t = f_t.reshape(batch_size, num_slots, -1)
            slots_tp1 = f_tp1.reshape(batch_size, num_slots, -1)
        except:
            assert False, "Embedding dim must be divisible by num_slots!!!"

        batch_size, num_slots, slot_len = slots_t.shape
        # logits: batch_size x num_slots x num_slots
        #        for each example (set of 8 slots), for each slot, dot product with every other slot at next time step
        logits = torch.matmul(self.score_fxn_3(slots_t),
                              slots_tp1.transpose(2,1))
        inp = logits.reshape(batch_size * num_slots, -1)
        target = torch.cat([torch.arange(num_slots) for i in range(batch_size)]).to(self.device)
        loss3 = nn.CrossEntropyLoss()(inp, target)
        acc3 = calculate_accuracy(inp.detach().cpu().numpy(), target.detach().cpu().numpy())
        if self.training:
            self.wandb.log({"tr_acc_struc": acc3})
            self.wandb.log({"tr_loss_struc": loss3.item()})
        else:
            self.wandb.log({"val_acc_struc": acc3})
            self.wandb.log({"val_loss_struc": loss3.item()})
        return loss3




