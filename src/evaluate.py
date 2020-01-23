import torch
from src.utils import EarlyStopping
from copy import deepcopy
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score as compute_f1_score
import warnings

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.utils import calculate_f1_score
"""Usage:
 tr_eps.extend(val_eps)
 tr_labels.extend(val_labels)
 f_tr, y_tr = get_feature_vectors(tr_eps, tr_labels)
 f_test, y_test = get_feature_vectors(test_eps, test_labels)
 trainer = SKLearnProbeTrainer(encoder=encoder)
 test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels, val_labels,
                                            test_eps, test_labels) """



class LinearAttentionProbe(nn.Module):
    """Attention oover slots to linear classifier"""
    def __init__(self, slot_len, num_classes):
        super().__init__()
        self.slot_len = slot_len
        self.attn = nn.MultiheadAttention(embed_dim=slot_len, num_heads=1)
        self.fc = nn.Linear(slot_len, num_classes)

    def forward(self, slots):
        slots = slots.transpose(1,0)
        try:
            v, w = self.attn(slots, slots, slots)
        except:
            print("hey")
        # just use the first token
        inp = v[0]
        w = w[:,0]
        unnormalized_logits = self.fc(inp)
        return unnormalized_logits, w

class MLPAttentionProbe(nn.Module):
    """Attention oover slots to linear classifier"""
    def __init__(self, slot_len, num_classes):
        super().__init__()
        self.slot_len = slot_len
        self.attn = nn.MultiheadAttention(embed_dim=slot_len, num_heads=1)
        self.mlp = nn.Sequential(nn.Linear(slot_len, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, num_classes)
                                  )

    def forward(self, slots):
        slots = slots.transpose(1,0)
        v, w = self.attn(slots, slots, slots)
        # just use the first token
        inp = v[0]
        w = w[:, 0]
        unnormalized_logits = self.mlp(inp)
        return unnormalized_logits, w




class AttentionProbeTrainer(object):
    def __init__(self, epochs,
                 encoder=None,
                 patience=15,
                 batch_size=64,
                 lr=1e-4,
                 type_="linear",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_classes = 256,
                 **kwargs):

        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.type_=type_
        self.device = device
        self.num_classes = num_classes
        self.encoder = encoder


    def fit_predict(self, x_tr, yt, x_val, yv, x_test, yte):
        if self.type_ == "linear":
            attn_probe = LinearAttentionProbe(slot_len=self.encoder.slot_len, num_classes=self.num_classes)
        else:
            attn_probe = MLPAttentionProbe(slot_len=self.encoder.slot_len, num_classes=self.num_classes)

        attn_probe.to(self.device)
        opt = torch.optim.Adam(list(attn_probe.parameters()), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=False)
        tr_ds, val_ds, test_ds = TensorDataset(x_tr, yt), TensorDataset(x_val, yv), TensorDataset(x_test, yte)
        tr_dl, val_dl, test_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True),\
                                 DataLoader(val_ds,batch_size=self.batch_size, shuffle=True), \
                                 DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)


        epoch = 0
        while (not early_stopper.early_stop) and epoch < self.epochs:
            attn_probe.train()
            for x, y in tr_dl:
                x = x.float() / 255.
                x = x.to(self.device)
                y = y.to(self.device)
                slots = self.encoder(x).detach()
                opt.zero_grad()
                pred, w = attn_probe(slots)
                loss = nn.CrossEntropyLoss()(pred, y.long())
                loss.backward()
                opt.step()
            attn_probe.eval()
            for x, y in val_dl:
                x = x.float() / 255.
                x = x.to(self.device)
                y = y.to(self.device)
                slots = self.encoder(x).detach()
                pred, w = attn_probe(slots)
                epoch_val_f1 = calculate_f1_score(pred.detach().cpu().numpy(), y.detach().cpu().numpy())
            early_stopper(epoch_val_f1, attn_probe)
            epoch += 1

        for x, y in test_dl:
            x = x.float() / 255.
            x = x.to(self.device)
            y = y.to(self.device)
            slots = self.encoder(x).detach()
            pred, w = attn_probe(slots)
            test_f1 = calculate_f1_score(pred.detach().cpu().numpy(), y.detach().cpu().numpy())
        avg_weight = w.mean(dim=0)
        return test_f1, avg_weight

    def train_test(self, x_tr, x_val, y_tr, y_val, x_test, y_test):
        print("train-test started!")
        f1_dict = {}
        weights_dict = {}
        for label_name in y_tr.keys():
            # print(label_name)
            yt = torch.tensor(y_tr[label_name]).long()
            yv = torch.tensor(y_val[label_name]).long()
            yte = torch.tensor(y_test[label_name]).long()
            test_f1, avg_weight = self.fit_predict(x_tr, yt, x_val, yv, x_test, yte)
            f1_dict[label_name] = test_f1
            weights_dict[label_name] = avg_weight.detach().cpu().numpy()

        importances_df = pd.DataFrame(weights_dict)
        # acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        return f1_dict, importances_df


















