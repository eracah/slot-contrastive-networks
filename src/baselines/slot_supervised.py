import torch
import torch.nn as nn
import numpy as np
from src.utils import EarlyStopping, appendabledict, append_suffix
from src.encoders import SlotEncoder
from src.utils import calculate_accuracy
from torch.utils.data import DataLoader, TensorDataset


class SupervisedTrainer():
    def __init__(self, args, device=torch.device('cpu'), wandb=None):
        super().__init__()

        self.args = args
        self.wandb = wandb
        self.patience = self.args.patience
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = device


        # bad convention, but these get set in "create_slot_classifiers"
        self.encoder = self.slot_fcs = self.early_stoppers = self.optimizers = None

    def create_slot_classifiers(self, sample_label):
        num_state_vars = len(sample_label)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, name="sup_loss")


        self.encoder = SlotEncoder(self.args.num_channels,self.args.slot_len, num_slots=num_state_vars, args= self.args).to(self.device)

        self.slot_fcs = []
        self.slot_fc_params = []
        for _ in range(self.encoder.num_slots):
            slot_fc = nn.Linear(self.encoder.slot_len, 256).to(self.device)
            self.slot_fcs.append(slot_fc)
            self.slot_fc_params.extend(list(slot_fc.parameters()))

        self.optimizer = torch.optim.Adam(list(self.slot_fc_params) + list(self.encoder.parameters()),
                                          lr=self.args.lr, eps=1e-5)

    def generate_batch(self, frames, label_dict, batch_size):
        label_values = [torch.tensor(labels) for labels in list(label_dict.values()) ]
        ds = TensorDataset(frames, *label_values)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for x, *ys in dl:
            labels = [y.long().to(self.device) for y in ys]
            yield x.float().to(self.device) / 255., labels

    def do_one_epoch(self, episodes, labels_dict):
        label_keys = list(labels_dict.keys())
        mode = "train" if self.encoder.training else "val"
        losses = []
        accs = []
        loss_slots = appendabledict()
        acc_slots = appendabledict()
        data_generator = self.generate_batch(episodes, labels_dict, batch_size=self.batch_size)
        for x, labels in data_generator:
            loss = 0.
            self.optimizer.zero_grad()
            slots = self.encoder(x)
            for i in range(self.encoder.num_slots):
                label_name = label_keys[i]
                slot_i = slots[:, i]
                slot_fc = self.slot_fcs[i]
                logits = slot_fc(slot_i)
                ground_truth = labels[i]
                loss_i = nn.CrossEntropyLoss()(logits, ground_truth)
                loss += loss_i
                _, acc_binary = calculate_accuracy(logits.detach().cpu().numpy(), ground_truth.detach().cpu().numpy())
                accs.extend(acc_binary)
                acc_slots.extend_update({label_name: acc_binary})
                loss_slots.append_update({label_name: loss_i.detach().cpu().numpy()})
            losses.append(loss.detach().cpu().numpy())


            if mode == "train":
                loss.backward()
                self.optimizer.step()

        epoch_loss = np.mean(losses)
        epoch_acc = np.mean(accs)

        loss_slots = {k: np.mean(v) for k,v in loss_slots.items()}
        acc_slots = {k: np.mean(v) for k,v in acc_slots.items()}

        self.log_results(epoch_loss, epoch_acc, loss_slots, acc_slots, mode=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss, self.encoder)


        return epoch_loss, epoch_acc, loss_slots, acc_slots

    def train(self, tr_eps, tr_labels, val_eps, val_labels):
        self.create_slot_classifiers(tr_labels)
        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch))
            self.encoder.train()
            for slot_fc in self.slot_fcs:
                slot_fc.train()
            tr_loss, tr_acc, tr_loss_slots, tr_acc_slots = self.do_one_epoch(tr_eps, tr_labels)

            self.wandb.log(append_suffix(tr_loss_slots,"_tr_loss"), step=epoch)
            self.wandb.log(append_suffix(tr_acc_slots,"_tr_acc"), step=epoch)

            self.encoder.eval()
            for slot_fc in self.slot_fcs:
                slot_fc.eval()
            val_loss, val_acc, val_loss_slots, val_acc_slots = self.do_one_epoch(val_eps, val_labels)
            self.wandb.log(append_suffix(val_loss_slots, "_val_loss"), step=epoch)
            self.wandb.log(append_suffix(val_acc_slots, "_val_acc"), step=epoch)
            self.wandb.log(dict(tr_loss=tr_loss, val_loss=val_loss, tr_acc=tr_acc, val_acc=val_acc), step=epoch)
            if self.early_stopper.early_stop:
                break
        # torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.args.env_name + '.pt'))
        return self.encoder

    def test(self, test_eps, test_labels):
        self.encoder.eval()
        for slot_fc in self.slot_fcs:
            slot_fc.eval()
        test_loss, test_acc, test_loss_slots, test_acc_slots = self.do_one_epoch(test_eps, test_labels)
        return test_acc, test_acc_slots


    def log_results(self, loss, acc, loss_slots, acc_slots, mode=""):
        print("\t{}: ".format(mode))
        print("\t\tLoss: {0:.4f}".format(loss))
        for label_name, loss in loss_slots.items():
            print("\t\t\t {}".format(label_name) + " Loss: {0:.4f}".format(loss))
        print("\t\tAcc: {0:.2f}%".format(100 * acc))
        for label_name, acc in acc_slots.items():
            print("\t\t\t {}".format(label_name) + " Acc: {0:.2f} %".format(100*acc))


        # self.wandb.log({prefix + '_loss': epoch_loss}, step=epoch_idx)
