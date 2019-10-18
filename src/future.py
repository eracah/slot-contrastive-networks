import torch
from atariari.benchmark.utils import appendabledict
from copy import deepcopy
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from atariari.benchmark.probe import postprocess_raw_metrics
from sklearn.metrics import f1_score as compute_f1_score
import warnings

"""Usage:
 tr_eps.extend(val_eps)
 tr_labels.extend(val_labels)
 f_tr, y_tr = get_feature_vectors(tr_eps, tr_labels)
 f_test, y_test = get_feature_vectors(test_eps, test_labels)
 trainer = SKLearnProbeTrainer(encoder=encoder)
 test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels, val_labels,
                                            test_eps, test_labels) """

def get_feature_vectors(encoder, episodes, episode_labels):
    vectors = []
    labels = appendabledict()
    for ep_ind in range(len(episodes)):
        x = torch.stack(episodes[ep_ind]).cpu() / 255.
        y = episode_labels[ep_ind]
        for label in y:
            labels.append_update(label)
        z = encoder(x).detach().cpu().numpy()
        vectors.append(z)
    vectors = np.concatenate(vectors)
    return vectors, labels


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



class SKLearnProbeTrainer(object):
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
        acc_dict, f1_dict = {}, {}
        for label_name in y_tr.keys():
            print(label_name)
            tr_labels = y_tr[label_name]
            test_labels = y_test[label_name]
            x_tr = deepcopy(f_tr)

            # sklearn is annoying about classes that only appear once or twice
            inds = [i for i, v in enumerate(tr_labels) if 0.2 * tr_labels.count(v) >= 3]
            tr_labels = [tr_labels[ind] for ind in inds]
            x_tr = np.stack([x_tr[ind] for ind in inds])
            self.estimator.fit(x_tr, tr_labels )
            y_pred = self.estimator.predict(f_test)
            accuracy, _ = calculate_accuracy(y_pred, test_labels, argmax=False)
            # if label_name == "player_x":
            #     print("hey")
            warnings.filterwarnings('ignore')
            f1score = compute_f1_score(test_labels, y_pred,  average="weighted")
            print("\t Acc: {}\n\t f1: {}".format(accuracy, f1score))
            acc_dict[label_name] = accuracy
            f1_dict[label_name] = f1score

            # reset estimator
            self.reset_estimator()

        acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        return acc_dict, f1_dict