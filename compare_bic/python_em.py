# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:51:41 2019

@author: Thomas Athey
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import os,sys
from bic import processBIC

"""
Reads in the various hierarchical agglomerations and performs GMM,
then calculates BIC and saves these values
"""
    
os.chdir('..')
X = np.genfromtxt('embedded_right.csv',delimiter=',',skip_header=1)
n = X.shape[0]
c_true = np.genfromtxt('classes.csv',skip_header=1)
os.chdir('./compare_bic')

r_files = ['EEE','EII','VII','VVV']
r_files = [('r_hc_' + r_files[i] + '.csv') for i in np.arange(len(r_files))]
python_files = ['average','complete','single','ward']
python_files = [('python_hc_' + python_files[i] + '.csv') for i in np.arange(len(python_files))]

covs = ['full','tied','diag','spherical']
files = r_files+python_files
kmax = 19
ks = np.arange(1,kmax+1)

for a,cov in enumerate(covs):
    bics = np.zeros((len(files),len(ks)))
    for b,file in enumerate(files):
        agglom = np.genfromtxt(file,skip_header=1,delimiter=',')
        for c,k in enumerate(ks):
            agglom_clusters = agglom[:,kmax-k]
            unqs = np.unique(agglom_clusters)
            label=0
            agglom_clusters_temp = agglom_clusters
            for un in unqs:
                agglom_clusters[agglom_clusters_temp==un] = label
                label += 1
            #sklearn GaussianMixture can initialize from certain distribution
            #parameters, but CANNOT initialize from cluster assignments
            #Therefore, we must calculate initial parameters from assignments
            one_hot = np.zeros([n,k])
            one_hot[np.arange(n),agglom_clusters.astype('int')] = 1
            weights, means, covariances = _estimate_gaussian_parameters(
                    X,one_hot,1e-06,cov)
            weights /=n
            precisions_cholesky_ = _compute_precision_cholesky(covariances,cov)
            if cov=='tied':
                ch = precisions_cholesky_
                precisions = np.dot(ch,ch.T)
            elif cov=='diag':
                precisions = precisions_cholesky_
            else:
                precisions = [np.dot(ch,ch.T) for ch in precisions_cholesky_]
            try: #try without regularization
                gmm = GaussianMixture(
                                n_components=k,covariance_type=cov,weights_init=weights,
                                means_init=means,precisions_init=precisions,
                                tol=1e-3,max_iter=100,reg_covar=0)
                c_hat = gmm.fit_predict(X)
            except ValueError: #if GMM doesn't work, start over and regularize 
                gmm = GaussianMixture(
                            n_components=k,covariance_type=cov,weights_init=weights,
                            means_init=means,precisions_init=precisions,
                            tol=1e-3,max_iter=100)
                c_hat = gmm.fit_predict(X)
            bic = processBIC(X,gmm.weights_,gmm.means_,gmm.covariances_,gmm.covariance_type)
            bics[b,c] = bic
    fname = 'python_bic_' + cov + '.csv'
    np.savetxt(fname,bics,delimiter=',',header='placeholder')
    
