import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from src.utils import Cutout
from src.trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
from src.encoders import SlotEncoder


class BilinearScoreFunction(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class NCETrainer(Trainer):
    def __init__(self, args, device=torch.device('cpu'), wandb=None):
        super().__init__(wandb, device)
        self.encoder = SlotEncoder(args.obs_space[0], args).to(device)
        self.args = args
        self.wandb = wandb
        self.patience = self.args.patience
        self.score_fxn = BilinearScoreFunction(self.encoder.slot_len, self.encoder.slot_len).to(device)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.score_fxn.parameters()) + list(self.encoder.parameters()),
                                          lr=args.lr, eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x, x_pos, x_neg = [], [], []
            for episode in episodes_batch:
                t, t_neg = np.random.randint(0, len(episode) - 1), np.random.randint(0, len(episode))
                t_pos = t + 1
                x.append(episode[t])
                x_pos.append(episode[t_pos])
                x_neg.append(episode[t_neg])
            yield torch.stack(x).to(self.device) / 255.,\
                  torch.stack(x_pos).to(self.device) / 255., \
                  torch.stack(x_neg).to(self.device) / 255.,


    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.score_fxn.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x, x_pos, x_neg in data_generator:
            slots_t, slots_pos, slots_neg = self.encoder(x),\
                                            self.encoder(x_pos),\
                                            self.encoder(x_neg)

            """Loss 1: Does a pair of slot vectors from the same slot
                       come from consecutive (or within a small window) time steps or not?"""
            loss1 = []
            for i in range(self.encoder.num_slots):
                # batch_size, slot_len
                slot_t_i, slot_pos_i, slot_neg_i = slots_t[:,i],\
                                                   slots_pos[:,i],\
                                                   slots_neg[:,i]

                # batch_size x 1
                pos_logits = self.score_fxn(slot_t_i, slot_pos_i)
                neg_logits = self.score_fxn(slot_t_i, slot_neg_i)

                logits = torch.cat((pos_logits, neg_logits), dim=1)

                # binary classification problem: from time t+1 or not
                # answer is always 0 b/c the first logit is always the positive correct one
                ground_truth = torch.zeros_like(pos_logits).long().squeeze()

                loss1i = nn.CrossEntropyLoss()(logits, ground_truth)
                loss1.append(loss1i)

            loss1 = np.sum(loss1)

            """Loss 2:  Does a pair of vectors close in time come from the 
                        same slot of different slots"""
            # pass for now
            loss2 = 0.0

            loss = loss1 # + loss2

            self.optimizer.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.encoder.train(), self.score_fxn.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.score_fxn.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.args.env_name + '.pt'))
        return self.encoder

    def log_results(self, epoch_idx, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss}, step=epoch_idx)
