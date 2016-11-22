# random_forest_cluster
A clustering algorithm, where a forest of shallow trees are trained on random subsets of the features. Points are clustered together according to how often they end up on the same leaf.

###Pseudocode:



Initialize data_transformed as empty array

Repeat n_iter times:

>  Select n_feat features at random
	
>  Train a decision tree regressor of depth max_depth
	
>  Add to data_transformed a new column with each data point's leaf assignment
	
Caclulate Jaccard distance matrix from data_transformed

Pass the distance matrix to k medoids
