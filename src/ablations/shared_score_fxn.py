import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from src.trainer import Trainer
from src.utils import EarlyStopping
from src.encoders import SlotEncoder
from src.evaluation.probe_modules import calculate_accuracy

import time


class BilinearScoreFunction(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class SharedScoreFxnTrainer(Trainer):
    def __init__(self, args, device=torch.device('cpu'), wandb=None):
        super().__init__(wandb, device)
        self.encoder = SlotEncoder(args.obs_space[0], args.slot_len, args.num_slots, args).to(device)
        self.args = args
        self.wandb = wandb
        self.patience = self.args.patience
        self.score_fxn = BilinearScoreFunction(self.encoder.slot_len, self.encoder.slot_len).to(device)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.score_fxn.parameters()) + list(self.encoder.parameters()),
                                          lr=args.lr, eps=1e-5)
        self.early_stopper1 = EarlyStopping(patience=self.patience, verbose=False, name="nce_loss1",
                                            savedir=self.wandb.run.dir + "/models")
        self.early_stopper2 = EarlyStopping(patience=self.patience, verbose=False, name="nce_loss2",
                                            savedir=self.wandb.run.dir + "/models")

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        # print('Total Steps: {}'.format(total_steps))
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
            yield torch.stack(x).to(self.device) / 255., \
                  torch.stack(x_pos).to(self.device) / 255., \
                  torch.stack(x_neg).to(self.device) / 255.,

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.score_fxn.training and self.score_fxn.training else "val"
        losses, loss1s, loss2s = [], [], []
        accs, acc1s, acc2s = [], [], []
        data_generator = self.generate_batch(episodes)
        t0 = time.time()
        loss1_slots = {"loss1_slot_{}".format(slot_num): [] for slot_num in range(self.encoder.num_slots)}
        loss2_slots = {"loss2_slot_{}".format(slot_num): [] for slot_num in range(self.encoder.num_slots)}
        acc1_slots = {"acc1_slot_{}".format(slot_num): [] for slot_num in range(self.encoder.num_slots)}
        acc2_slots = {"acc2_slot_{}".format(slot_num): [] for slot_num in range(self.encoder.num_slots)}
        for iteration, (x, x_pos, x_neg) in enumerate(data_generator):
            slots_t, slots_pos, slots_neg = self.encoder(x), \
                                            self.encoder(x_pos), \
                                            self.encoder(x_neg)

            """Loss 1: Does a pair of slot vectors from the same slot
                       come from consecutive (or within a small window) time steps or not?"""
            loss1is = []
            t10 = time.time()
            for i in range(self.encoder.num_slots):
                # batch_size, slot_len
                slot_t_i, slot_pos_i, slot_neg_i = slots_t[:, i], \
                                                   slots_pos[:, i], \
                                                   slots_neg[:, i]

                # batch_size x 1
                pos_logits = self.score_fxn(slot_t_i, slot_pos_i)
                neg_logits = self.score_fxn(slot_t_i, slot_neg_i)

                logits = torch.cat((pos_logits, neg_logits), dim=1)

                # binary classification problem: from time t+1 or not
                # answer is always 0 b/c the first logit is always the positive correct one
                ground_truth = torch.zeros_like(pos_logits).long().squeeze().to(self.device)

                loss1i = nn.CrossEntropyLoss()(logits, ground_truth)
                _, acc1i = calculate_accuracy(logits.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
                loss1_slots["loss1_slot_{}".format(i)].append(loss1i.detach().cpu().numpy())
                acc1_slots["acc1_slot_{}".format(i)].append(acc1i)
                loss1is.append(loss1i)
                acc1s.extend(acc1i)
                accs.extend(acc1i)

            loss1 = torch.mean(torch.stack(loss1is))
            loss1s.append(loss1.detach().cpu().numpy())

            # self.log_results(iteration, loss1, type_="iteration", prefix=mode)
            # print("\t loss1 iter time {} ".format(time.time() - t10))
            # self.wandb.log({"loss1": loss1}, step=iteration)

            """Loss 2:  Does a pair of vectors close in time come from the 
                        same slot of different slots"""
            loss2is = []
            t20 = time.time()
            for i in range(self.encoder.num_slots):
                batch_size = slot_t_i.shape[0]
                other_slot_inds = list(range(self.encoder.num_slots))
                random_other_slot_index = np.random.choice(other_slot_inds)
                other_slot_inds.remove(i)
                slot_t_i = slots_t[:, i]
                pos_logit = self.score_fxn(slot_t_i, slots_pos[:, i])
                neg_logit = self.score_fxn(slot_t_i, slots_pos[:, random_other_slot_index])
                logits = [pos_logit, neg_logit]
                # correct label in this binary classification problem is always 0 b/c pos logit comes first
                ground_truth = torch.zeros((batch_size,)).long().to(self.device)
                logits = torch.cat(logits, dim=1)
                loss2i = nn.CrossEntropyLoss()(logits, ground_truth)
                _, acc2i = calculate_accuracy(logits.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
                loss2_slots["loss2_slot_{}".format(i)].append(loss2i.detach().cpu().numpy())
                acc2_slots["acc2_slot_{}".format(i)].append(acc2i)
                acc2s.extend(acc2i)
                accs.extend(acc2i)
                loss2is.append(loss2i)

            loss2 = torch.mean(torch.stack(loss2is))
            loss2s.append(loss2.detach().cpu().numpy())

            loss = self.args.loss1_coeff * loss1 + loss2
            losses.append(loss.detach().cpu().numpy())

            self.optimizer.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()

        epoch_loss = np.mean(losses)
        epoch_loss1 = np.mean(loss1s)
        epoch_loss2 = np.mean(loss2s)
        epoch_loss1_slots = {k: np.mean(v) for k, v in loss1_slots.items()}
        epoch_loss2_slots = {k: np.mean(v) for k, v in loss2_slots.items()}
        loss_terms = {mode + "_loss1": epoch_loss1, mode + "_loss2": epoch_loss2}
        other_losses = {}
        other_losses.update(epoch_loss1_slots)
        other_losses.update(epoch_loss2_slots)
        other_losses = {mode + "_" + k: v for k, v in other_losses.items()}

        epoch_acc = np.mean(accs)
        epoch_acc1 = np.mean(acc1s)
        epoch_acc2 = np.mean(acc2s)
        epoch_acc1_slots = {k: np.mean(v) for k, v in acc1_slots.items()}
        epoch_acc2_slots = {k: np.mean(v) for k, v in acc2_slots.items()}
        acc_terms = {mode + "_acc1": epoch_acc1, mode + "_acc2": epoch_acc2}
        other_accs = {}
        other_accs.update(epoch_acc1_slots)
        other_accs.update(epoch_acc2_slots)
        other_accs = {mode + "_" + k: v for k, v in other_accs.items()}

        self.log_results(epoch_loss, epoch_loss1, epoch_loss2, epoch_acc, epoch_acc1, epoch_acc2, mode=mode)
        print("\t\tnum iterations: {}".format(iteration + 1))
        print("\t\tepoch time: {}".format(time.time() - t0))
        if mode == "val":
            self.early_stopper1(-epoch_loss1, self.encoder)
            self.early_stopper2(-epoch_loss2, self.encoder)

        return epoch_loss, other_losses, epoch_acc, other_accs, acc_terms, loss_terms

    def train(self, tr_eps, val_eps):
        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch))
            self.encoder.train(), self.score_fxn.train(), self.score_fxn.train()
            tr_loss, other_tr_losses, tr_acc, other_tr_accs, tr_acc_terms, tr_loss_terms = self.do_one_epoch(epoch,
                                                                                                             tr_eps)
            self.wandb.log(tr_loss_terms, step=epoch)
            self.wandb.log(tr_acc_terms, step=epoch)
            # self.wandb.log(other_tr_losses, step=epoch)
            # self.wandb.log(other_tr_accs, step=epoch)

            self.encoder.eval(), self.score_fxn.eval(), self.score_fxn.eval()
            val_loss, other_val_losses, val_acc, other_val_accs, val_acc_terms, val_loss_terms = self.do_one_epoch(
                epoch, val_eps)
            self.wandb.log(val_loss_terms, step=epoch)
            self.wandb.log(val_acc_terms, step=epoch)
            # self.wandb.log(other_val_losses, step=epoch)
            # self.wandb.log(other_val_accs, step=epoch)

            self.wandb.log(dict(tr_loss=tr_loss, val_loss=val_loss, tr_acc=tr_acc, val_acc=val_acc), step=epoch)
            if self.early_stopper1.early_stop and self.early_stopper2.early_stop:
                break
        # torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.args.env_name + '.pt'))
        return self.encoder

    def log_results(self, loss, loss1, loss2, acc, acc1, acc2, mode=""):
        print("\t{}: ".format(mode))
        print("\t\tLoss: {0:.4f}".format(loss))
        print("\t\t\tLoss1: {0:.4f}".format(loss1))
        print("\t\t\tLoss2: {0:.4f}".format(loss2))
        print("\t\tAcc: {0:.2f}%".format(100 * acc))
        print("\t\t\tAcc1: {0:.2f}%".format(100 * acc1))
        print("\t\t\tAcc2: {0:.2f}%".format(100 * acc2))

        # self.wandb.log({prefix + '_loss': epoch_loss}, step=epoch_idx)