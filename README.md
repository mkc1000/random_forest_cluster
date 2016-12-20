# random_forest_cluster
Intra-feature Random Forest Clustering (IRFC) is an algorithm wherein a forest of shallow trees is trained on random subsets of the features (each tree trained on a different random subset). Points are clustered together according to how often they end up on the same leaf. IRFC is preliminarily the best available clustering algorithm per Yeung's ([2001](http://bioinformatics.oxfordjournals.org/content/17/4/309.short)) FOM cluster validation metric. Further details provided in [the paper introducing the algorithm](https://github.com/mkc1000/random_forest_cluster/blob/master/Intra-Feature_Random_Forest_Clustering.pdf).

###Pseudocode:


>  Initialize data_transformed as empty array

>  Repeat n_iter times:

>  >  Select n_feat features at random from data
	
>  >  Train a decision tree regressor of depth max_depth
	
>  >  Add to data_transformed a new column with each data point's leaf assignment
	
>  Caclulate Jaccard distance matrix between the rows of data_transformed (each row represents a point's leaf assignments)

>  Pass the distance matrix to k medoids, which returns cluster assignments


###Example Usage:

    from sklearn.datasets import load_boston
    from rf_cluster import RFCluster
    
    X = load_boston().data
    rfc = RFCluster(k=5)
    clusters = rfc.fit_predict(X)

###Validating Clustering Algorithms, Example Usage: (See Yeung, 2001)

    from sklearn.datasets import load_boston
    from rf_cluster import RFCluster
    from sklearn.cluster import KMeans
    from within_cluster_variance import WCVScore
   
    X = load_boston().data
    kmeans = KMeans(n_clusters=5)
    rfc = RFCluster(k=5)
    wcv_kmeans = WCVScore(kmeans)  # the model passed to WCVScore must have a fit_predict method
    wcv_rfc = WCVScore(rfc)
    kmeans_score, kmeans_nclusters = wcv_kmeans.score(X)
    rfc_score, rfc_nclusters = wcv_rfc.score(X)  # warning: this line takes a few minutes

A lower score indicates a better algorithm. Two algorithms are only comparable per this metric when they output the same number of clusters, because when a clustering algorithm outputs a greater number of clusters, it is trivially easier to achieve a lower score.

The following plot compares IRFC (using the default parameters) with k-means on the boston housing dataset. The output of the code above is plotted, as n_clusters is varied from 5 to other values.
