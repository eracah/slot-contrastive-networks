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

        self.classifier1 = nn.Linear(self.emnedding_dim, self.fmap_encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.fmap_encoder.local_layer_depth, self.fmap_encoder.local_layer_depth).to(device)
        self.device = device

    def calc_loss(self, xt, a, xtp1):
        f_t = self.encoder(xt)
        fmap_t = self.fmap_encoder(xt).permute(0, 2, 3, 1)
        fmap_tp1 = self.fmap_encoder(xtp1).permute(0, 2, 3, 1)
        # Loss 1: Global at time t, f5 patches at time t-1
        sy = fmap_tp1.size(1)
        sx = fmap_tp1.size(2)
        N = f_t.size(0)
        loss1 = 0.
        acc1 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(f_t)
                positive = fmap_tp1[:, y, x, :]
                logits = torch.matmul(predictions, positive.t())
                target = torch.arange(N).to(self.device)
                step_loss = F.cross_entropy(logits, target)
                acc1 += calculate_accuracy(logits.detach().cpu().numpy(), target.detach().cpu().numpy())
                loss1 += step_loss
        loss1 = loss1 / (sx * sy)
        acc1 = acc1 / (sx * sy)

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        loss2 = 0.
        acc2 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier2(fmap_t[:, y, x, :])
                positive = fmap_tp1[:, y, x, :]
                logits = torch.matmul(predictions, positive.t())
                target = torch.arange(N).to(self.device)
                step_loss = F.cross_entropy(logits, target)
                acc2 += calculate_accuracy(logits.detach().cpu().numpy(), target.detach().cpu().numpy())
                loss2 += step_loss
        loss2 = loss2 / (sx * sy)
        acc2 = acc2 / (sx * sy)
        loss = loss1 + loss2
        if self.training:
            self.wandb.log({"tr_acc1": acc1})
            self.wandb.log({"tr_acc2": acc2})
            self.wandb.log({"tr_loss1": loss1})
            self.wandb.log({"tr_loss2": loss2})
        else:
            self.wandb.log({"val_acc1": acc1})
            self.wandb.log({"val_acc2": acc2})
            self.wandb.log({"val_loss1": loss1})
            self.wandb.log({"val_loss2": loss2})
        return loss


