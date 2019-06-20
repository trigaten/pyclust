import pytest
import numpy as np
import sys
sys.path.append("C:\\Users\\Thomas Athey\\Documents\\Labs\\Labs\\jovo\\clustering\\pyclust")
from compare_bic.bic import processBIC
from sklearn.mixture import GaussianMixture
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.metrics import adjusted_rand_score
from brute_cluster_experiments import brute_cluster



def test_biccalc():
    """
    Tests our BIC calculation for all four models on a set of random data
    """
    covs = ['full','tied','diag','spherical']
    X = np.random.multivariate_normal(mean=[0,0,0],cov=np.identity(3),size=100)
    for cov in covs:
        gmm = GaussianMixture(n_components=2, covariance_type=cov)
        gmm.fit(X)
        bic1 = processBIC(X,gmm.weights_,gmm.means_,gmm.covariances_,cov)
        bic2 = gmm.bic(X)
        assert_almost_equal(-bic1,bic2)

def test_clustering():
    """
    Tests that our 'none' option for agglomeration is identical to a normal scikit-learn GaussianMixture
    Because there is randomness in GMM's initialization, the test data should be somewhat clusterable to expect the same results
    """
    np.random.seed(seed=1)
    covs = ['full','spherical','tied','diag']
    x = np.random.uniform(size=[100,3])
    x2 = np.random.uniform(low=2.0,high=3.0,size=[100,3])
    x = np.append(x,x2,axis=0)
    for cov in covs:
        gmm = GaussianMixture(n_components=2, covariance_type=cov)
        c_hat_gmm1 = gmm.fit_predict(x)
        #gmm2 = GaussianMixture(n_components=2, covariance_type=cov)
        #c_hat_gmm2 = gmm2.fit_predict(x)
        #print(adjusted_rand_score(c_hat_gmm1,c_hat_gmm2))

        c_hat_brute,_,_ = brute_cluster(x, ['none'], ['none'], [cov], [2])
        ari = adjusted_rand_score(c_hat_gmm1,c_hat_brute)
        assert_almost_equal(ari,1.0)

test_clustering()