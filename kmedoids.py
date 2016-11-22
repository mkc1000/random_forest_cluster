import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mutual_info_score
from scipy.sparse.linalg import eigs
from numpy.linalg import matrix_power
from joblib import Parallel, delayed

def jaccard(x,y):
    return np.mean(x!=y)

def weighted_jaccard(x,y,w=None):
    """w is list of weights, same length as x and y summing to 1"""
    if w is None:
        return jaccard(x,y)
    return (x!=y).dot(w)

def jaccard_distance_matrix(X, n_jobs=1):
    vint = np.vectorize(int)
    X_int = vint(X*100)
    return pairwise_distances(X_int, metric=jaccard, n_jobs=n_jobs)

def weighted_jaccard_distance_matrix(X, w, n_jobs=1):
    """w has length X.shape[1]"""
    vint = np.vectorize(int)
    X_int = vint(X*100)
    distance_matrix = pairwise_distances(X_int, w=w, metric=weighted_jaccard,n_jobs=n_jobs)
    return distance_matrix

class JKMedoids(object):
    def __init__(self, k, max_iter=100, n_attempts=10, accepting_weights=True, weight_adjustment=0, n_jobs=1):
        self.k = k
        self.n_attempts = n_attempts
        self.n_jobs=n_jobs
        if max_iter is None:
            self.max_iter = -1
        else:
            self.max_iter = max_iter
        self.accepting_weights = accepting_weights
        self.weight_adjustment = weight_adjustment
        self.adjusting_weights = self.weight_adjustment != 0
        self.distance_matrix = None
        self.assignments = None
        self.assignment_score = None
        self.weights = None

    def fit_once(self, X):
        assignments = np.random.randint(0,self.k,size=X.shape[0])
        old_assignments = np.zeros(assignments.shape)
        it = 0
        while (old_assignments != assignments).any() and it != self.max_iter:
            it += 1
            old_assignments = assignments

            centroids = []
            for cluster in xrange(self.k):
                mask = assignments == cluster
                if np.sum(mask) == 0:
                    continue
                within_cluster_distance_matrix = (self.distance_matrix[mask]).T
                most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
                centroids.append(most_central_point)

            to_centroid_distance_matrix = (self.distance_matrix[centroids]).T
            assignments = np.apply_along_axis(np.argmin, 1, to_centroid_distance_matrix)

            if self.adjusting_weights:
                weight_update = np.apply_along_axis(lambda col: mutual_info_score(assignments, col), 0, X)
                weight_update = weight_update/np.sum(weight_update)
                self.weights = self.weights*(1-self.weight_adjustment) + weight_update*self.weight_adjustment
                self.distance_matrix = weighted_jaccard_distance_matrix(X, self.weights)
        return assignments

    def score(self, assignments):
        centroids = []
        for cluster in xrange(self.k):
            mask = assignments == cluster
            if np.sum(mask) == 0:
                continue
            within_cluster_distance_matrix = (self.distance_matrix[mask]).T
            most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
            centroids.append(most_central_point)
        to_centroid_distance_matrix = (self.distance_matrix[centroids]).T
        scores = np.apply_along_axis(np.min, 1, to_centroid_distance_matrix)
        score = np.sum(scores)
        return score

    def fit(self, X):
        if self.accepting_weights:
            X, self.weights = X
            self.distance_matrix = weighted_jaccard_distance_matrix(X, self.weights, self.n_jobs)
        else:
            self.distance_matrix = jaccard_distance_matrix(X, self.n_jobs)
        for _ in xrange(self.n_attempts):
            assignments = self.fit_once(X)
            if self.assignments is None:
                self.assignments = assignments
                self.assignment_score = self.score(self.assignments)
            else:
                score = self.score(assignments)
                if score < self.assignment_score:
                    self.assignment_score = score
                    self.assignments = assignments
        return self

    def fit_predict(self, X, _=None):
        self.fit(X)
        return self.assignments

class SquishyJKMedoids(object):
    def __init__(self, k, max_iter=100, n_attempts=10, accepting_weights=True, weight_adjustment=0, n_jobs=1):
        self.k = k
        self.n_attempts = n_attempts
        if max_iter is None:
            self.max_iter = -1
        else:
            self.max_iter = max_iter
        self.accepting_weights = accepting_weights
        self.distance_matrix = None
        self.to_centroid_distances = None
        self.assignments = None
        self.assignment_score = None
        self.n_jobs=n_jobs

    def fit_once(self, X):
        assignments = np.random.randint(0,self.k,size=X.shape[0])
        old_assignments = np.zeros(assignments.shape)
        it = 0
        while (old_assignments != assignments).any() and it != self.max_iter:
            it += 1
            old_assignments = assignments

            self.to_centroid_distances = []
            centroids = []
            first_run = True
            for cluster in xrange(self.k):
                mask = assignments == cluster
                if np.sum(mask) == 0:
                    continue
                if first_run:
                    distance_matrix = self.distance_matrix
                else:
                    weights = np.apply_along_axis(lambda col: mutual_info_score(mask, col), 0, X)
                    weights = weights/np.sum(weights)
                    distance_matrix = weighted_jaccard_distance_matrix(X, weights, n_jobs=self.n_jobs)
                within_cluster_distance_matrix = (distance_matrix[mask]).T
                most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
                centroids.append(most_central_point)
                self.to_centroid_distances.append(distance_matrix[:,most_central_point].reshape(-1))

            to_centroid_distance_matrix = (np.array(self.to_centroid_distances)).T
            assignments = np.apply_along_axis(np.argmin, 1, to_centroid_distance_matrix)

            first_run = False
        return assignments

    def score(self, assignments):
        centroids = []
        for cluster in xrange(self.k):
            mask = assignments == cluster
            if np.sum(mask) == 0:
                continue
            within_cluster_distance_matrix = (self.distance_matrix[mask]).T
            most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
            centroids.append(most_central_point)
        to_centroid_distance_matrix = (self.distance_matrix[centroids]).T
        scores = np.apply_along_axis(np.min, 1, to_centroid_distance_matrix)
        score = np.sum(scores)
        return score

    def fit(self, X):
        if self.accepting_weights:
            X, self.weights = X
        else:
            self.weights = np.ones(X.shape[1])/X.shape[1]
        self.distance_matrix = weighted_jaccard_distance_matrix(X, self.weights, n_jobs=self.n_jobs)
        for _ in xrange(self.n_attempts):
            assignments = self.fit_once(X)
            if self.assignments is None:
                self.assignments = assignments
                self.assignment_score = self.score(self.assignments)
            else:
                score = self.score(assignments)
                if score < self.assignment_score:
                    self.assignment_score = score
                    self.assignments = assignments
        return self

    def fit_predict(self, X, _=None):
        self.fit(X)
        return self.assignments
