import numpy as np
from rf_transform import RFTransform
from eig_weighting import EigenvectorWeighting
from kmedoids import JKMedoids, SquishyJKMedoids
from sklearn.pipeline import Pipeline

class RFCluster(Pipeline):
    def __init__(self, k,
                model_type='random_forest',
                kmedoids_type='normal',
                n_forests=150,
                n_trees=1,
                n_features_to_predict=0.5,
                max_depth=5, #should be 2 if model_type is boosting
                learning_rate=0.6,
                using_weights=False,
                using_pca=False,
                weight_extent=1, # 2 if model_type is boosting
                max_iter=60,
                n_attempts=10,
                weight_adjustment=0,
                eig_extent=0,
                n_jobs=1):
        rft = RFTransform(n_forests,
                        model_type=model_type,
                        n_trees=n_trees,
                        n_features_to_predict=n_features_to_predict,
                        max_depth=max_depth,
                        outputting_weights=using_weights,
                        using_pca=using_pca,
                        weight_extent=weight_extent,
                        learning_rate=learning_rate,
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
