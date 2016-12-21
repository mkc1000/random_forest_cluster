import numpy as np
from rf_transform import RFTransform
from eig_weighting import EigenvectorWeighting
from kmedoids import JKMedoids, SquishyJKMedoids
from sklearn.pipeline import Pipeline

class RFCluster(Pipeline):
    """
    A clustering algorithm wherein a forest of shallow trees is trained on random subsets of the features (each tree trained
    on a different random subset). Points are clustered together according to how often they end up on the same leaf.
    
    Parameters:
    
      k: Int. Number of clusters to partition the data into. (Note: for large k, RFCluster often returns fewer than k clusters.
            This occurs in the final k-medoids step. If, at any given step, one data point is selected as the next centroid for
            two clusters, this reduces the number of clusters returned.
      
      n_trees: Int. Number of decision trees in the ensemble.
      
      n_features_to_predict: Float, int, or string. Determines how many features form the target for each decision tree.
            If float, takes that fraction of the total number of features (i.e., if there are 20 features in
            the data, and n_features_to_predict = 0.4, each decision tree will be trained to predict the
            values of 8 features, given the remaining 12.
            If int, that many features will be predicted with each decision tree.
            If 'sqrt', n_features_to_predict will take on the square root of the total number of features.
            If 'log', same as for 'sqrt', but using the log base 2 instead.
      
      max_depth: Int. The maximum depth of each decision tree.
      
      using_weights: Boolean. If True, when ensembling the partitions from each decision tree, some trees will have extra
            weight, given the variance of the features that the decision trees are trained on.
      
      weight_extent: Nonnegative float. Only relevant if using_weights.
      
      max_iter: Int. Maximum number of iterations in the k-medoids step.
      
      n_attempts: Int. Number of attempts for k-medoids. After n_attempts attempts, the partition with the lowest within-
            cluster-sum-of-squares is selected.
      
      k_medoids_type: String. If "normal," standard k-medoids is employed. If "minkowski", Minkowski weighted k-medoids is
            used. (This is unrelated to the Minkowski distance metric).
      
      weight_adjustment: Float. Only relevant if using_weights. The extent to which weights can be adjusted during the
            k-medoids.
      
      eig_extent: Nonnegative int. Only relevant if using_weights. Partitions gain more weight when they have more mutual
            information with other partitions. As eig_extent goes to infinity, this becomes the only criterion.
                  
      using_pca: Boolean. If True, PCA is applied to the data first.
    
    Methods:
    
    fit_predict:
        Input: 2D numpy array, each row a data point, each column a feature
        Output: 1D numpy array with cluster assignments for each data point
      
    """
    def __init__(self, k,
                n_trees=150,
                n_features_to_predict=0.5,
                max_depth=5,
                using_weights=False,
                weight_extent=1,
                max_iter=60,
                n_attempts=10,
                kmedoids_type='normal',
                weight_adjustment=0,
                eig_extent=0,
                using_pca=False,
                n_jobs=1):
        rft = RFTransform(n_trees,
                        n_features_to_predict=n_features_to_predict,
                        max_depth=max_depth,
                        outputting_weights=using_weights,
                        using_pca=using_pca,
                        weight_extent=weight_extent,
                        n_jobs=n_jobs)
        ew = EigenvectorWeighting(extent=eig_extent)
        if kmedoids_type == 'normal':
            jk = JKMedoids(k,
                            max_iter=max_iter,
                            n_attempts=n_attempts,
                            accepting_weights=using_weights,
                            weight_adjustment=weight_adjustment,
                            n_jobs=n_jobs)
        else:
            jk = SquishyJKMedoids(k,
                            max_iter=max_iter,
                            n_attempts=n_attempts,
                            accepting_weights=using_weights,
                            weight_adjustment=weight_adjustment,
                            n_jobs=n_jobs)
        if eig_extent == 0 or not using_weights:
            Pipeline.__init__(self,[('rft', rft), ('jkmeans', jk)])
        else:
            Pipeline.__init__(self,[('rft', rft), ('ew', ew), ('jkmeans', jk)])
