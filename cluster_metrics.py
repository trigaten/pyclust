import numpy as np
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score


def best_supercluster(c1,c2):
    """
    inputs:
    c1 - nx1 array with membership values for clustering with a higher cluster number
    c2 - nx1 array other clustering
    outputs:
    ari - float best ARI from combining subclusters
    """
    vals = np.unique(c1)
    c_new = np.zeros(len(c1))
    for v in vals:
        modes,_ = mode(c2[c1==v])
        c_new[c1==v] = modes[0]
    return adjusted_rand_score(c2, c_new), c_new
