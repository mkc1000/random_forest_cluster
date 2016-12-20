import numpy as np
from joblib import Parallel, delayed
import random
from datetime import datetime

def mean_cluster_variances(clusters, feature):
    """Inputs:
         clusters: 1D numpy array of cluster IDs, shape (n)
         feature: 1D numpy array values for a given feature, shape (n)
        
       Output: Float; the weighted average across all clusters of the within-cluster-variance of the input feature
    """
    unique_clusters = np.unique(clusters)
    variances = []
    sizes = []
    for cluster in unique_clusters:
        cluster_feature = feature[clusters == cluster]
        variances.append(np.var(cluster_feature))
        sizes.append(len(cluster_feature))
    weighted_average_variance = np.array(variances).dot(np.array(sizes)) / len(feature)
    return weighted_average_variance

def score_once(wcv, data, i):
    """Effectively a method to the object below, but pulled outside so it could parallelized more easily.
    Inputs:
      wcv: A WCVScore model; acts just like "self"
      data: 2D numpy array
      i: Int; column index for data
      
    Output: Float; the within-cluster-variance score only for feature i,
            given cluster assignments that are generated only using the remaining features
    """
    y = data[:, i]
    X = np.delete(data, i, axis=1)
    predictions = wcv.model.fit_predict(X)
    n_clusters = len(np.unique(predictions))
    within_cluster_variance = mean_cluster_variances(predictions, y)
    total_variance = np.var(y)
    scaled_within_cluster_variance = within_cluster_variance / total_variance
    return scaled_within_cluster_variance, n_clusters

class WCVScore(object):
    """
    Arguments:
      model: a clustering model with a fit_predict method. That fit_predict method must take a 2D numpy array, shape (n, k)
             and return a 1D numpy array of cluster IDs, shape (n)
      max_iter: Features are withheld one by one, and self.model.fit_predict is called each time.
                If max_iter < k, this will only be done for a random subset of the features.
      n_jobs: Allows parallelization of the loop described above.
      
    Methods:
      score:
        Inputs:
          data: 2D numpy array
        Output: (cluster validation score for the algorithm implemented in self.model,
                 average number of clusters identified by self.model)
    """
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
            features = np.random.choice(features, size=self.max_iter, replace=False)
        output = Parallel(n_jobs=self.n_jobs)(delayed(score_once)(self, data, i) for i in features)
        output = np.array(output)
        self.wcvs = output[:, 0]
        self.n_clusters = output[:, 1]
        return np.mean(self.wcvs), np.mean(self.n_clusters)
