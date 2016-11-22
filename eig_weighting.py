import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mutual_info_score
from scipy.sparse.linalg import eigs
from numpy.linalg import matrix_power
from joblib import Parallel, delayed

class EigenvectorWeighting(object):
    def __init__(self, complete=False, extent=1):
        self.data = None
        self.weights = None
        self.complete = complete
        self.extent = extent

    def fit(self, data):
        data, weights = data
        self.data = data
        mutual_info_matrix = pairwise_distances(data.T, metric=mutual_info_score)
        if self.complete:
            evals, evecs = eigs(mutual_info_matrix, k=1)
            weights = evecs.reshape(-1)
            self.weights = weights/np.sum(weights)
        else:
            weights = weights.reshape(-1,1)
            mutual_info_power = matrix_power(mutual_info_matrix, self.extent)
            weights = mutual_info_power.dot(weights)
            weights = weights.reshape(-1)
            self.weights = weights/np.sum(weights)

    def transform(self, data, _=None):
        self.fit(data)
        return self.data, self.weights

    def fit_transform(self, data, _=None):
        self.fit(data)
        return self.data, self.weights
