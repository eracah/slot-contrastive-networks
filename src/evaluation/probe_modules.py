import torch
from src.utils import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.utils import calculate_f1_score
import sys
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression


"""Usage:
 tr_eps.extend(val_eps)
 tr_labels.extend(val_labels)
 f_tr, y_tr = get_feature_vectors(tr_eps, tr_labels)
 f_test, y_test = get_feature_vectors(test_eps, test_labels)
 trainer = SKLearnProbeTrainer(encoder=encoder)
 test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels, val_labels,
                                            test_eps, test_labels) """
import torch
from torch import nn
from src.utils import append_suffix, compute_dict_average, calculate_multiple_accuracies, calculate_multiple_f1_scores
import numpy as np
import sys
class ProbeTrainer(object):
    def __init__(self,
                 encoder = None,
                 wandb=None,
                 method_name = "my_method",
                 patience = 15,
                 num_classes = 256,
                 num_state_variables=8,
                 fully_supervised = False,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr = 5e-4,
                 epochs = 100,
                 batch_size = 64,
                 representation_len = 256,
                 num_slots =1,
                 per_probe_early_stop=True,
                 l1_regularization=False):

        self.encoder = encoder.to(device)
        self.wandb = wandb
        self.l1_regularization = l1_regularization
        self.num_state_variables = num_state_variables
        self.device = device
        self.fully_supervised = fully_supervised
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.per_probe_early_stop = per_probe_early_stop
        self.num_slots = num_slots
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.representation_len = representation_len
        if self.encoder == None:
            self.vector_input = True
        else:
            self.vector_input = False

        self.probe = nn.Linear(self.representation_len, self.num_classes * self.num_state_variables).to(self.device)

        if self.per_probe_early_stop:
            self.early_stoppers = [ [ EarlyStopping(patience=self.patience, name="slot_%i_label_%i"%(s, l)) for s in range(self.num_slots)  ]\
                                    for l in range(self.num_state_variables)  ]
        else:
            self.early_stoppers = EarlyStopping(patience=self.patience,name="total_loss")

        parameters = list(self.probe.parameters())
        if self.fully_supervised:
            parameters += self.encoder.parameters()
        self.optimizer = torch.optim.Adam(parameters,eps=1e-5, lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.2, verbose=True, mode='max', min_lr=1e-5)


    def do_probe(self, x):
        if self.vector_input:
            vectors = x
        elif self.fully_supervised:
            vectors = self.encoder(x)
        else:
            vectors = self.encoder(x).detach()

        batch_size, *rest = vectors.shape
        preds = self.probe(vectors)
        preds = preds.reshape(batch_size, -1, self.num_state_variables, self.num_classes)
        preds = preds.transpose(1, 3)

        return preds


    def do_one_epoch(self, dataloader):
        losses = []
        all_preds = []
        all_labels = []
        for x,y in dataloader:
            frames = x.float().to(self.device) / 255.
            labels = y.long().to(self.device)
            preds = self.do_probe(frames)
            lbls = labels[:, :, None].repeat(1, 1, preds.shape[3]) # for if preds has another dimension for slot encoders for example
            loss_terms = nn.CrossEntropyLoss(reduction="none")(preds, lbls)
            if self.probe.training and self.per_probe_early_stop:
                lt = []
                for sv in range(self.num_state_variables):
                    for sl in range(self.num_slots):
                        if self.early_stoppers[sv][sl].early_stop:
                            loss_term = loss_terms[:,sv, sl].detach()
                        else:
                            loss_term = loss_terms[:, sv, sl]
                        lt.append(loss_term)
                loss_terms = torch.stack(lt)
            loss = loss_terms.mean()

            # if self.l1_regularization:
            #     l1_norm = torch.norm(self.probe.weight, p=1)
            #     loss = loss + l1_norm
            if self.probe.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            preds = preds.cpu().detach().numpy().squeeze()
            preds = np.argmax(preds, axis=1).reshape(preds.shape[0], -1)
            all_preds.append(preds)
            lbls = lbls.cpu().detach().numpy()
            all_labels.append(lbls.reshape(lbls.shape[0], -1))
            losses.append(loss.detach().item())

        epoch_loss = np.mean(losses)

        labels_tensor = np.concatenate(all_labels)
        preds_tensor = np.concatenate(all_preds)
        accuracies = calculate_multiple_accuracies(preds_tensor, labels_tensor)
        f1_scores = calculate_multiple_f1_scores(preds_tensor, labels_tensor)

        return epoch_loss, accuracies, f1_scores



    def train(self, tr_dl, val_dl):
        self.encoder.to(self.device)
        epoch = 0

        while epoch < self.epochs:
            es = []
            self.probe.train()
            epoch_loss, accuracies, _ = self.do_one_epoch(tr_dl)

            self.probe.eval()
            val_loss, val_accuracies, _ = self.do_one_epoch(val_dl)
            val_accuracies = np.asarray(val_accuracies).reshape(self.num_state_variables, self.num_slots)
            if self.per_probe_early_stop:
                for sv in range(self.num_state_variables):
                    for sl in range(self.num_slots):
                        self.early_stoppers[sv][sl](val_accuracies[sv, sl])
                        es.append(self.early_stoppers[sv][sl].early_stop)
            else:
                self.early_stoppers(np.mean(val_accuracies))
                es.append(self.early_stoppers.early_stop)
            self.log_results(epoch, tr_loss=epoch_loss, val_loss=val_loss)
            epoch += 1
            if all(es):
                break
        sys.stderr.write("Probe done!\n")

        torch.save(self.probe.state_dict(), self.wandb.run.dir + "/probe.pt")

    def test(self, test_dl):
        self.probe.eval()
        _, acc, f1 = self.do_one_epoch(test_dl)
        return f1

    def get_weights(self):
        return self.probe.weight.detach().cpu().numpy()

    def log_results(self, epoch_idx, **kwargs):
        sys.stderr.write("Epoch: {}\n".format(epoch_idx))
        for k, v in kwargs.items():
            if isinstance(v, dict):
                sys.stderr.write("\t {}:\n".format(k))
                for kk, vv in v.items():
                    sys.stderr.write("\t\t {}: {:8.4f}\n".format(kk, vv))
                    self.wandb.log({kk:vv})
                sys.stderr.write("\t ------\n")

            else:
                sys.stderr.write("\t {}: {:8.4f}\n".format(k, v))
                self.wandb.log({k: v})


def get_feature_vectors(encoder, dataloader):
    num_state_variables = dataloader.dataset.tensors[1].shape[-1]
    vectors = []
    labels = np.empty(shape=(0, num_state_variables))
    for x,y in dataloader:
        frames = x.float() / 255.
        h = encoder(frames).detach().cpu().numpy()
        vectors.append(h)
        labels = np.concatenate((labels, y))
    vectors = np.concatenate(vectors)
    return vectors, labels

class LinearRegressionProbe(object):
    def __init__(self,
                 encoder):
        self.encoder = encoder
        self.multi_lin_reg = MultiOutputRegressor(LinearRegression())


    def get_weights(self):
        return np.stack([estimator.coef_ for estimator in self.multi_lin_reg.estimators_])

    def train(self, tr_dl, val_dl):
        x, y = get_feature_vectors(self.encoder.cpu(), tr_dl)
        self.multi_lin_reg.fit(x,y)

    def test(self, test_dl):
        x, y = get_feature_vectors(self.encoder.cpu(), test_dl)
        r2_scores = [self.multi_lin_reg.estimators_[i].score(x,y[:,i]) for i in range(y.shape[-1])]
        return r2_scores


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
        #early_stopper = EarlyStopping(patience=self.patience, verbose=False)
        tr_ds, val_ds, test_ds = TensorDataset(x_tr, yt), TensorDataset(x_val, yv), TensorDataset(x_test, yte)
        tr_dl, val_dl, test_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True),\
                                 DataLoader(val_ds,batch_size=self.batch_size, shuffle=True), \
                                 DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)


        epoch = 0
        while epoch < self.epochs:
            losses = []
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
                losses.append(loss.detach().cpu().numpy())
            sys.stderr.write("Epoch {}:\n".format(epoch))
            sys.stderr.write("\t Train Loss {}:\n".format(np.mean(losses)))

            attn_probe.eval()
            losses = []
            for x, y in val_dl:
                x = x.float() / 255.
                x = x.to(self.device)
                y = y.to(self.device)
                slots = self.encoder(x).detach()
                pred, w = attn_probe(slots)
                loss = nn.CrossEntropyLoss()(pred, y.long())
                losses.append(loss.detach().cpu().numpy())
            sys.stderr.write("\t Val Loss {}:\n".format(np.mean(losses)))
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


















