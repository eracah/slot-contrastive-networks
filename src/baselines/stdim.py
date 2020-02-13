import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils import calculate_accuracy

class STDIMModel(nn.Module):
    def __init__(self, encoder, embedding_dim, config, device=torch.device('cpu'), wandb=None):
        super().__init__()
        self.encoder = encoder
        self.fmap_encoder = encoder.fmap_encoder
        self.config = config
        self.emnedding_dim = embedding_dim
        self.wandb = wandb

        self.score_fxn1 = nn.Linear(self.emnedding_dim, self.fmap_encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.score_fxn2 = nn.Linear(self.fmap_encoder.local_layer_depth, self.fmap_encoder.local_layer_depth).to(device)
        self.device = device

    def calc_loss(self, xt, a, xtp1):
        f_t = self.encoder(xt)
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


