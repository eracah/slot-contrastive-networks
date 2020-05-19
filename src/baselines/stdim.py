import torch.nn as nn
import torch
from src.utils import calculate_accuracy

class STDIMModel(nn.Module):
    def __init__(self, encoder, args, global_vector_len, device=torch.device('cpu'), wandb=None):
        super().__init__()
        self.encoder = encoder
        self.fmap_encoder = encoder.get_f5
        self.global_vector_len = global_vector_len
        self.wandb = wandb
        self.args = args

        self.score_fxn1 = nn.Linear(self.global_vector_len, self.encoder.local_vector_len) # x1 = global, x2=patch, n_channels = 32
        self.score_fxn2 = nn.Linear(self.encoder.local_vector_len, self.encoder.local_vector_len)
        self.device = device

    def calc_global_to_local(self, global_t, local_tp1):
        N, sy, sx, d = local_tp1.shape
        # Loss 1: Global at time t, f5 patches at time t+1
        glob_score = self.score_fxn1(global_t)
        local_flattened = local_tp1.reshape(-1, d)
        # [N*sy*sx, d] @  [d, N] = [N*sy*sx, N ] -> dot product of every global vector in batch with local voxel at all spatial locations for all examples in the batch
        # then reshape to sy*sx, N, N then to sy*sx*N, N
        logits1 = torch.matmul(local_flattened, glob_score.t()).reshape(N, sy * sx, -1).transpose(1, 0).reshape(-1, N)
        # we now have sy*sx N x N matrices where the diagonals correspond to dot product between pairs consecutive in time at the same bagtch index
        # aka the correct answer. So the correct logit index is the diagonal sx*sy times
        target1 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss1 = nn.CrossEntropyLoss()(logits1, target1)
        acc1 = calculate_accuracy(logits1.detach().cpu().numpy(), target1.detach().cpu().numpy())
        return loss1, acc1


    def calc_local_to_local(self, local_t, local_tp1):
        N, sy, sx, d = local_tp1.shape
        # Loss 2: f5 patches at time t, with f5 patches at time t+1
        local_t_score = self.score_fxn2(local_t.reshape(-1, d))
        transformed_local_t = local_t_score.reshape(N, sy*sx,d).transpose(0,1)
        local_tp1 = local_tp1.reshape(N, sy * sx, d).transpose(0, 1)
        logits2 = torch.matmul(transformed_local_t, local_tp1.transpose(1, 2)).reshape(-1, N)
        target2 = torch.arange(N).repeat(sx * sy).to(self.device)
        loss2 = nn.CrossEntropyLoss()(logits2, target2)
        acc2 = calculate_accuracy(logits2.detach().cpu().numpy(), target2.detach().cpu().numpy())
        return loss2, acc2

    def calc_loss(self, xt, a, xtp1):
        f_t = self.encoder(xt)
        f_tp1 = self.encoder(xtp1)
        fmap_t = self.fmap_encoder(xt).permute(0, 2, 3, 1)
        fmap_tp1 = self.fmap_encoder(xtp1).permute(0, 2, 3, 1)


        loss1, acc1 = self.calc_global_to_local(f_t, fmap_tp1)
        loss2, acc2 = self.calc_local_to_local(fmap_t, fmap_tp1)

        loss = loss1 + loss2

        for k,v in dict(loss1=loss1, loss2=loss2, acc1=acc1, acc2=acc2).items():
            self.log(k,v)
        return loss

    def log(self, name, scalar):
        if self.training:
            self.wandb.log({"tr_"+ name: scalar})
        else:
            self.wandb.log({"val_"+ name: scalar})



