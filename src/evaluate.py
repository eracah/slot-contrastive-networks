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
from torch.optim import Adam
"""Usage:
 tr_eps.extend(val_eps)
 tr_labels.extend(val_labels)
 f_tr, y_tr = get_feature_vectors(tr_eps, tr_labels)
 f_test, y_test = get_feature_vectors(test_eps, test_labels)
 trainer = SKLearnProbeTrainer(encoder=encoder)
 test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels, val_labels,
                                            test_eps, test_labels) """


def encode_feature_vectors(encoder, device, *frame_tensors):
    encoder.to(device)
    zs = []
    for frame_tensor in frame_tensors:
        x = frame_tensor.to(torch.float).to(device) / 255.
        z = encoder(x).detach().to(device)
        zs.append(z)
    return zs


def calculate_f1_score(logits, labels):
    preds = np.argmax(logits, axis=1)
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_accuracy(logits, labels, argmax=True):
    if argmax:
        preds = np.argmax(logits, axis=1)
    else:
        preds = logits
    correct_or_not = (preds == labels).astype(int)
    acc = np.mean(correct_or_not)
    return acc, correct_or_not



class LinearAttentionProbe(nn.Module):
    """Attention oover slots to linear classifier"""
    def __init__(self, slot_len, num_classes):
        super().__init__()
        self.slot_len = slot_len
        self.attn = nn.MultiheadAttention(embed_dim=slot_len, num_heads=1)
        self.fc = nn.Linear(slot_len, num_classes)

    def forward(self, slots):
        slots = slots.transpose(1,0)
        v, w = self.attn(slots, slots, slots)
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
        w = w[:,0]
        unnormalized_logits = self.mlp(inp)
        return unnormalized_logits, w




class AttentionProbeTrainer(object):
    def __init__(self, epochs, patience=15, batch_size=64, lr=1e-4, type="linear", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),**kwargs):

        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.type=type
        self.device = device


    def fit_predict(self, f_tr, yt, f_val, yv, f_test, yte):
        if type == "linear":
            attn_probe = LinearAttentionProbe(slot_len=f_tr.shape[2], num_classes=np.max(yt) + 1)
        else:
            attn_probe = MLPAttentionProbe(slot_len=f_tr.shape[2], num_classes=np.max(yt) + 1)

        attn_probe.to(self.device)
        opt = torch.optim.Adam(list(attn_probe.parameters()), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=False)
        f_tr, yt,f_val, yv,f_test, yte = torch.tensor(f_tr),\
                                         torch.tensor(yt),torch.tensor(f_val),\
                                         torch.tensor(yv), torch.tensor(f_test), torch.tensor(yte)
        tr_ds, val_ds, test_ds = TensorDataset(f_tr, yt), TensorDataset(f_val, yv), TensorDataset(f_test, yte)
        tr_dl, val_dl, test_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True),\
                                 DataLoader(val_ds,batch_size=f_val.shape[0], shuffle=True), \
                                 DataLoader(val_ds, batch_size=f_test.shape[0], shuffle=True)


        epoch = 0
        while (not early_stopper.early_stop) and epoch < self.epochs:
            attn_probe.train()
            for x, y in tr_dl:
                x.to(self.device)
                y.to(self.device)
                opt.zero_grad()
                pred, w = attn_probe(x)
                loss = nn.CrossEntropyLoss()(pred,y.long())
                #acc = calculate_accuracy(pred,y)
                loss.backward()
                opt.step()
            attn_probe.eval()
            for x, y in val_dl:
                pred, w = attn_probe(x)
                epoch_val_f1 = calculate_f1_score(pred.detach().numpy(), y.detach().numpy())
            early_stopper(epoch_val_f1, attn_probe)
            epoch += 1

        for x, y in test_dl:
            pred, w = attn_probe(x)
            test_f1 = calculate_f1_score(pred.detach().numpy(), y.detach().numpy())
        avg_weight = w.mean(dim=0)
        return test_f1, avg_weight

    def train_test(self, f_tr, y_tr, f_val, y_val, f_test, y_test):
        print("train-test started!")
        f1_dict = {}
        weights_dict = {}
        for label_name in y_tr.keys():
            # print(label_name)
            yt = y_tr[label_name]
            yv = y_val[label_name]
            yte = y_test[label_name]
            test_f1, avg_weight = self.fit_predict(f_tr, yt, f_val, yv, f_test, yte)
            f1_dict[label_name] = test_f1
            weights_dict[label_name] = avg_weight.detach().numpy()

        importances_df = pd.DataFrame(weights_dict)
        # acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        return f1_dict, importances_df



class MLPProbeTrainer(object):
    def __init__(self,
                 patience=15, **kwargs):

        self.patience = patience


        self.estimator = MLPClassifier(hidden_layer_sizes=(256,),
                                                    early_stopping=True,
                                                    n_iter_no_change=self.patience,
                                                    validation_fraction=0.2,
                                                    tol=1e-3)

    def reset_estimator(self):
        self.estimator = MLPClassifier(hidden_layer_sizes=(256,),
                                       early_stopping=True,
                                       n_iter_no_change=self.patience,
                                       validation_fraction=0.2,
                                       tol=1e-3)



    def train_test(self, f_tr, y_tr, f_test,y_test):
        print("train-test started!")
        f1_dict = {}
        weights_dict = {}
        for label_name in y_tr.keys():
            # print(label_name)
            tr_labels = y_tr[label_name]
            test_labels = y_test[label_name]
            x_tr = deepcopy(f_tr)

            # sklearn is annoying about classes that only appear once or twice
            inds = [i for i, v in enumerate(tr_labels) if 0.2 * tr_labels.count(v) >= 3]
            tr_labels = [tr_labels[ind] for ind in inds]
            x_tr = np.asarray([x_tr[ind] for ind in inds])
            self.estimator.fit(x_tr, tr_labels)
            y_pred = self.estimator.predict(f_test)
            # accuracy, _ = calculate_accuracy(y_pred, test_labels, argmax=False)
            warnings.filterwarnings('ignore')
            f1score = compute_f1_score(test_labels, y_pred, average="weighted")
            f1_dict[label_name] = f1score
            # weights = self.estimator.feature_importances_.T
            # weights_dict[label_name] = deepcopy(weights)
            # reset estimator
            self.reset_estimator()

        # acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        return f1_dict, weights_dict


class GBTProbeTrainer(object):
    def __init__(self,
                 patience=15,
                 num_processes=4,
                 lr=5e-4,
                 epochs=100, **kwargs):

        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.num_processes = num_processes


        self.estimator = GradientBoostingClassifier(
                                                    n_iter_no_change=self.patience,
                                                    validation_fraction=0.2,
                                                    tol=1e-3)

    def reset_estimator(self):
        self.estimator = GradientBoostingClassifier(
                                                    n_iter_no_change=self.patience,
                                                    validation_fraction=0.2,
                                                    tol=1e-3)



    def train_test(self, f_tr, y_tr, f_test,y_test):
        print("train-test started!")
        f1_dict = {}
        weights_dict = {}
        for label_name in y_tr.keys():
            # print(label_name)
            tr_labels = y_tr[label_name]
            test_labels = y_test[label_name]
            x_tr = deepcopy(f_tr)

            # sklearn is annoying about classes that only appear once or twice
            inds = [i for i, v in enumerate(tr_labels) if 0.2 * tr_labels.count(v) >= 3]
            tr_labels = [tr_labels[ind] for ind in inds]
            x_tr = np.asarray([x_tr[ind] for ind in inds])
            self.estimator.fit(x_tr, tr_labels)
            y_pred = self.estimator.predict(f_test)
            # accuracy, _ = calculate_accuracy(y_pred, test_labels, argmax=False)
            warnings.filterwarnings('ignore')
            f1score = compute_f1_score(test_labels, y_pred, average="weighted")
            f1_dict[label_name] = f1score
            weights = self.estimator.feature_importances_.T
            weights_dict[label_name] = deepcopy(weights)
            # reset estimator
            self.reset_estimator()

        # acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        return f1_dict, weights_dict






class LinearProbeTrainer(object):
    def __init__(self,
                 patience=15,
                 num_processes=4,
                 lr=5e-4,
                 epochs=100):

        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.num_processes = num_processes


        self.estimator = SGDClassifier(loss='log',
                                       n_jobs=self.num_processes,
                                       penalty='none',
                                       l1_ratio=0.0,
                                       alpha=0,
                                       learning_rate='constant',
                                       eta0=self.lr,
                                       max_iter=self.epochs,
                                       early_stopping=True,
                                       n_iter_no_change=self.patience,
                                       validation_fraction=0.2,
                                       tol=1e-3)

    def reset_estimator(self):
        self.estimator = SGDClassifier(loss='log',
                                       n_jobs=self.num_processes,
                                       penalty='none',
                                       l1_ratio=0.0,
                                       alpha=0,
                                       learning_rate='constant',
                                       eta0=self.lr,
                                       max_iter=self.epochs,
                                       early_stopping=True,
                                       n_iter_no_change=self.patience,
                                       validation_fraction=0.2,
                                       tol=1e-3)



    def train_test(self, f_tr, y_tr, f_test,y_test):
        print("train-test started!")
        f1_dict = {}
        weights_dict = {}
        for label_name in y_tr.keys():
            #print(label_name)
            tr_labels = y_tr[label_name]
            test_labels = y_test[label_name]
            x_tr = deepcopy(f_tr)

            # sklearn is annoying about classes that only appear once or twice
            inds = [i for i, v in enumerate(tr_labels) if 0.2 * tr_labels.count(v) >= 3]
            tr_labels = [tr_labels[ind] for ind in inds]
            x_tr = np.asarray([x_tr[ind] for ind in inds])
            self.estimator.fit(x_tr, tr_labels)
            y_pred = self.estimator.predict(f_test)
            #accuracy, _ = calculate_accuracy(y_pred, test_labels, argmax=False)
            warnings.filterwarnings('ignore')
            f1score = compute_f1_score(test_labels, y_pred,  average="weighted")
            f1_dict[label_name] = f1score
            weights = self.estimator.coef_.T
            weights_dict[label_name] = deepcopy(weights)
            # reset estimator
            self.reset_estimator()

        #acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)


        return f1_dict, weights_dict





