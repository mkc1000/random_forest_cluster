import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mutual_info_score
from scipy.sparse.linalg import eigs
from numpy.linalg import matrix_power
from joblib import Parallel, delayed


### Helper functions to be parallelized, if possible
def fit_rf_model(rftransform, X, i, features_to_predict):
    y_temp = X[:, features_to_predict]
    if rftransform.model_type == 'gradient_boosting':
        y_temp = np.apply_along_axis(np.mean,1,y_temp)
    X_temp = np.delete(X, features_to_predict, axis=1)
    rf_fit = rftransform.rfs[i].fit(X_temp, y_temp)
    return rf_fit

def get_predictions(rftransform, X, i, features_to_predict):
    y_temp = X[:, features_to_predict]
    X_temp = np.delete(X, features_to_predict, axis=1)
    predictions = rftransform.rfs[i].predict(X_temp)
    if len(predictions.shape) > 1:
        predictions = np.sum(predictions, 1)
    return predictions

def get_weight(rftransform, X, features_to_predict):
    y_temp = X[:, features_to_predict]
    y_temp_var = np.sum(np.apply_along_axis(np.var, 0, y_temp))
    weight = (1/y_temp_var)**rftransform.weight_extent
    return weight

class RFTransform(object):
    def __init__(self,
                 n_forests,
                 model_type='random_forest',
                 n_trees=1,
                 n_features_to_predict=0.5,
                 max_depth=5,
                 outputting_weights=True,
                 using_pca=True,
                 weight_extent=1,
                 learning_rate=0.9,
                 n_jobs=1):
        self.n_forests = n_forests
        self.n_trees = n_trees
        self.n_features_to_predict = n_features_to_predict
        self.outputting_weights = outputting_weights
        self.weight_extent = weight_extent
        self.n_jobs = n_jobs
        self.model_type = model_type
        if self.model_type == 'random_forest':
            self.rfs = [RandomForestRegressor(n_trees, max_depth=max_depth, n_jobs=-1) for _ in xrange(n_forests)]
        elif self.model_type == 'gradient_boosting':
            self.rfs = [GradientBoostingRegressor(n_estimators=n_trees, learning_rate=learning_rate, max_depth=max_depth) for _ in xrange(n_forests)]
        else:
            raise ValueError("RFTransform.model_type must be 'random_forest' or 'gradient_boosting'.")
        self.using_pca = using_pca
        if not self.using_pca and self.outputting_weights:
            print "Warning: It makes no sense to output weights if you're not using pca."
        self.pca = PCA()
        self.ss1 = StandardScaler()
        self.decision_paths = None
        self.features_indices = []
        if outputting_weights:
            self.weights = []

    def fit(self, X_init, *args, **kwargs):
        self.features_indices = []

        X_ss = self.ss1.fit_transform(X_init)
        if self.using_pca:
            X = self.pca.fit_transform(X_ss)
        else:
            X = X_ss
        if isinstance(self.n_features_to_predict, float):
            n_output = int(self.n_features_to_predict * X.shape[1])
        elif isinstance(self.n_features_to_predict, int):
            n_output = self.n_features_to_predict
        elif self.n_features_to_predict == 'sqrt':
            n_output = int(np.sqrt(X.shape[1]))
        elif self.n_features_to_predict == 'log':
            n_output = int(np.log2(X.shape[1]))

        if n_output == 0:
            n_output = 1

        for i in xrange(self.n_forests):
            features_to_predict = np.random.choice(np.arange(X.shape[1]),(n_output,),replace=False)
            self.features_indices.append(features_to_predict)

        self.rfs = Parallel(n_jobs=self.n_jobs)(delayed(fit_rf_model)(self, X, i, features_to_predict) for i, features_to_predict in enumerate(self.features_indices))
        return self

    def transform(self, X_init):
        self.decision_paths = None
        if self.outputting_weights:
            self.weights = []

        X_ss = self.ss1.transform(X_init)
        if self.using_pca:
            X = self.pca.transform(X_ss)
        else:
            X = X_ss

        decision_paths = Parallel(n_jobs=self.n_jobs)(delayed(get_predictions)(self,X,i,features_to_predict)
                                                      for i, features_to_predict in enumerate(self.features_indices))
        self.decision_paths = (np.array(decision_paths)).T

        if self.outputting_weights:
            self.weights = Parallel(n_jobs=self.n_jobs)(delayed(get_weight)(self,X,features_to_predict) for features_to_predict in self.features_indices)
            self.weights = np.array(self.weights)
            self.weights = self.weights/np.sum(self.weights)
            return self.decision_paths, self.weights
        else:
            return self.decision_paths

    def fit_transform(self, X_init, *args, **kwargs):
        self.fit(X_init)
        return self.transform(X_init)
