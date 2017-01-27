from sklearn.cluster import KMeans
import numpy as np

def run_kmeans(X, no_clusters):

    pred = KMeans(n_clusters=no_clusters, random_state=0).fit_predict(X)
#    print pred.labels_
#    print pred.cluster_centers_
    return pred
