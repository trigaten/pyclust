import pytest
import numpy as np
import sys
sys.path.append("C:\\Users\\Thomas Athey\\Documents\\Labs\\Labs\\jovo\\clustering\\pyclust")
from compare_bic.bic import processBIC
from sklearn.mixture import GaussianMixture
from numpy.testing import assert_almost_equal, assert_equal



def test_biccalc():
    covs = ['full','tied','diag','spherical']
    X = np.random.multivariate_normal(mean=[0,0,0],cov=np.identity(3),size=100)
    for cov in covs:
        gmm = GaussianMixture(n_components=2, covariance_type=cov)
        gmm.fit(X)
        bic1 = processBIC(X,gmm.weights_,gmm.means_,gmm.covariances_,cov)
        bic2 = gmm.bic(X)
        assert_almost_equal(-bic1,bic2)