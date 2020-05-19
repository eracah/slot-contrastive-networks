from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import sys

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

class ConcatRegressionProbe(object):
    def __init__(self,
                 encoder):
        self.encoder = encoder
        self.multi_lin_reg = None

    def train(self, tr_dl):
        x, y = get_feature_vectors(self.encoder.cpu(), tr_dl)
        self.multi_lin_reg.fit(x,y)

    def test(self, test_dl):
        x, y = get_feature_vectors(self.encoder.cpu(), test_dl)
        r2_scores = [self.multi_lin_reg.estimators_[i].score(x,y[:,i]) for i in range(y.shape[-1])]
        return r2_scores


class LinearRegressionProbe(ConcatRegressionProbe):
    def __init__(self,
                 encoder):
        super().__init__(encoder)
        self.multi_lin_reg = MultiOutputRegressor(LinearRegression())

    def get_feature_importances(self):
        return np.stack([estimator.coef_ for estimator in self.multi_lin_reg.estimators_])

class GBTRegressionProbe(ConcatRegressionProbe):
    def __init__(self,
                 encoder):
        super().__init__(encoder)
        self.multi_lin_reg = MultiOutputRegressor(GradientBoostingRegressor())

    def get_feature_importances(self):
        return np.stack([estimator.feature_importances_ for estimator in self.multi_lin_reg.estimators_])















