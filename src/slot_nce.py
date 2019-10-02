import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from src.trainer import Trainer
from src.utils import EarlyStopping
from src.encoders import SlotEncoder
import time

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
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, name="nce",
                                           savedir=self.wandb.run.dir + "/models")

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
        t0 = time.time()
        for iteration, (x, x_pos, x_neg) in enumerate(data_generator):
            slots_t, slots_pos, slots_neg = self.encoder(x),\
                                            self.encoder(x_pos),\
                                            self.encoder(x_neg)

            """Loss 1: Does a pair of slot vectors from the same slot
                       come from consecutive (or within a small window) time steps or not?"""
            loss1 = []
            t10 = time.time()
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
                ground_truth = torch.zeros_like(pos_logits).long().squeeze().to(self.device)

                loss1i = nn.CrossEntropyLoss()(logits, ground_truth)
                loss1.append(loss1i)


            loss1 = np.sum(loss1)
            self.log_results(iteration, loss1, type_="iteration", prefix=mode)
            print("\t loss1 iter time {} ".format(time.time() - t10))
            #self.wandb.log({"loss1": loss1}, step=iteration)


            """Loss 2:  Does a pair of vectors close in time come from the 
                        same slot of different slots"""
            loss2 = []
            t20 = time.time()
            for i in range(self.encoder.num_slots):
                slot_t_i = slots_t[:,i]
                logits = []
                batch_size = slot_t_i.shape[0]
                ground_truth = i * torch.ones((batch_size,)).long().to(self.device)
                for j in range(self.encoder.num_slots):
                    slot_pos_j = slots_pos[:,j]
                    # when i = j this is a postive logit
                    logit = self.score_fxn(slot_t_i, slot_pos_j)
                    logits.append(logit)
                logits = torch.cat(logits,dim=1)
                loss2i = nn.CrossEntropyLoss()(logits, ground_truth)
                loss2.append(loss2i)



            loss2 = np.sum(loss2)
            self.log_results(iteration, loss2, type_="iteration", prefix=mode)
            print("\t loss2 iter time {} ".format(time.time() - t20))
            #self.wandb.log({"loss2": loss2}, step=iteration)

            loss = loss1 + loss2

            self.optimizer.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            steps += 1
            epoch_loss += loss.detach().item() / steps

        print("num iterations: {}".format(iteration + 1))
        print("epoch time: {}".format(time.time() - t0))
        self.log_results(epoch, epoch_loss, type_="Epoch", prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss, self.encoder)
        return epoch_loss

    def train(self, tr_eps, val_eps):
        for epoch in range(self.epochs):
            self.encoder.train(), self.score_fxn.train()
            tr_loss = self.do_one_epoch(epoch, tr_eps)

            self.encoder.eval(), self.score_fxn.eval()
            val_loss = self.do_one_epoch(epoch, val_eps)

            self.wandb.log(dict(tr_loss=tr_loss, val_loss=val_loss), step=epoch)
            if self.early_stopper.early_stop:
                break
        #torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.args.env_name + '.pt'))
        return self.encoder

    def log_results(self, idx, loss, type_="Epoch", prefix=""):
        print("{} {}: {}, {} Loss: {}".format(prefix.capitalize(), type_, type_, idx, loss))

        #self.wandb.log({prefix + '_loss': epoch_loss}, step=epoch_idx)
