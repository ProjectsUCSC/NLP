from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

def lsa(data):
    
    #X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
    print "Running LSA"
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    return svd.fit_transform(data) 
#    print(svd.explained_variance_ratio_) 
#    print(svd.explained_variance_ratio_.sum())
#    return data
