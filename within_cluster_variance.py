import numpy as np
from joblib import Parallel, delayed
import random
from datetime import datetime

def mean_cluster_variances(clusters, feature):
    unique_clusters = np.unique(clusters)
    variances = []
    sizes = []
    for cluster in unique_clusters:
        cluster_feature = feature[clusters==cluster]
        variances.append(np.var(cluster_feature))
        sizes.append(len(cluster_feature))
    return np.array(variances).dot(np.array(sizes))/len(feature)

def score_once(wcv, data, i):
    y = data[:,i]
    X = np.delete(data, i, axis=1)
    predictions = wcv.model.fit_predict(X)
    n_clusters = len(np.unique(predictions))
    within_cluster_variance = mean_cluster_variances(predictions, y)
    total_variance = np.var(y)
    scaled_within_cluster_variance = within_cluster_variance / total_variance
    return scaled_within_cluster_variance, n_clusters

class WCVScore(object):
    def __init__(self, model, max_iter=10, n_jobs=1):
        self.model = model
        self.wcvs = []
        self.n_clusters = []
        self.n_jobs=n_jobs
        self.max_iter = max_iter

    def score(self, data):
        self.wcvs = []
        n_features = data.shape[1]
        features = np.arange(n_features)
        if n_features > self.max_iter:
            features = np.random.choice(features,size=self.max_iter,replace=False)
        output = Parallel(n_jobs=self.n_jobs)(delayed(score_once)(self, data, i) for i in features)
        output = np.array(output)
        self.wcvs = output[:,0]
        self.n_clusters = output[:,1]
        return np.mean(self.wcvs), np.mean(self.n_clusters)
