# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:12:14 2019

@author: Thomas Athey
"""

import numpy as np
from scipy.stats import multivariate_normal

"""
Calculates likelihood of a set of data from a GMM, then calculates BIC
Inputs:
    x - nxd datapoints
    wts - list of mixture weights (same length as means and variances)
    means - list of d numpy arrays of mixture means
    variances - list of dxd covariance matrices
    k - number of parameters
    
Outputs:
    bic - BIC where higher is better
"""
def calcBIC(x,wts,means,variances,k):
    n = x.shape[0]
    likelihood = 0
    for wt,mu,var in zip(wts,means,variances):
        mu = np.squeeze(mu)
        var = np.squeeze(var)
        var = multivariate_normal(mu,var)
        likelihood += wt*var.pdf(x)
    loglik = np.sum(np.log(likelihood))
    bic = 2*loglik-np.log(n)*k
    return bic


"""
Calculates BIC from input that is formatted either as the sklearn GaussianMixture
components or from data that was saved to a csv in R
 
Inputs
    data - nxd numpy array of data
    wts - k numpy array of mixture weights
    mus - kxd numpy array of means
    covs - kxdxd in the case of r and in python, the shape depends on the model type
        (see GaussianMixture class)
    m - a string that specifies the model, and implies that format of the other inputs
        (e.g. 'VII' implies that the parameters were read from a csv that was written by R)
Outputs
    BIC - bic value as calculated by the function above
"""
def processBIC(data,wts,mus,covs,m):
    d = data.shape[1]
    k = len(wts)
    
    #These options indicate mclust model types, so the format of covs is how
    #it was written to a csv in R
    if m == 'VII':
        params = k*(1+d+1)
        covs = np.split(covs,covs.shape[0])
    elif m == 'EEE':
        params = k*(1+d)+d*(d+1)/2
        covs = np.split(covs,covs.shape[0])
    elif m == 'VVV':
        params = k*(1+d+d*(d+1)/2)
        covs = np.split(covs,covs.shape[0])
    elif m == 'VVI':
        params = k*(1+d+d)
        covs = np.split(covs,covs.shape[0])
    #These options indicate GaussianMixture types, so the format of covs is 
    #sklearrn.mixture.GaussianMixture.covariances_
    elif m == 'spherical':
        params = k*(1+d+1)
        covs = [v*np.identity(d) for v in covs]
    elif m == 'tied':
        params = k*(1+d)+d*(d+1)/2
        covs = [covs for v in np.arange(k)]
    elif m == 'full':
        params = k*(1+d+d*(d+1)/2)
        covs = np.split(covs,covs.shape[0])
    elif m == 'diag':
        params = k*(1+d+d)
        covs = [np.diag(covs[i,:]) for i in np.arange(k)]
        
    params = params-1 #because the weights must add to 1
    wts = np.split(wts,wts.shape[0])
    means = np.split(mus,mus.shape[0])
    return calcBIC(data,wts,means,covs,params)