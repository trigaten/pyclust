# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:50:27 2019

Reads the parameter results from mclust's GMM and calculates BIC

@author: Thomas Athey
"""
import numpy as np
import os
from bic import processBIC

os.chdir('..')
X = np.genfromtxt('embedded_right.csv',delimiter=',',skip_header=1)
n = X.shape[0]
c_true = np.genfromtxt('classes.csv',skip_header=1)

os.chdir('./compare_bic/r_em_params')

r_files = ['EEE','EII','VII','VVV']
python_files = ['average','complete','single','ward']

covs = ['VVV','EEE','VVI','VII']
files = r_files+python_files
kmax = 19
ks = np.arange(1,kmax+1)

for a,cov in enumerate(covs):
    bics = np.zeros((len(files),len(ks)))
    for b,file in enumerate(files):
        for c,k in enumerate(ks):
            fname = 'r_' + cov + '_' + file + '_' + str(k) + '_'
            try:
                weights = np.genfromtxt(fname + 'weights.csv',delimiter=',',skip_header=1)
                means = np.genfromtxt(fname + 'means.csv',delimiter=',',skip_header=1).T
                if k==1:
                    weights = np.array([weights])
                    means = np.expand_dims(means,axis=0)
                covariances = np.genfromtxt(fname + 'variances.csv',delimiter=',',skip_header=1)
                covariances = np.split(covariances,k,axis=1)
                covariances = [np.expand_dims(covariances[i],axis=0) for i in np.arange(len(covariances))]
                covariances = np.concatenate(covariances)
                bic = processBIC(X,weights,means,covariances,cov)
                bics[b,c] = bic
    
            except IOError:
                    continue
    fname = 'r_bic_' + cov + '.csv'
    os.chdir('..')
    np.savetxt(fname,bics,delimiter=',',header='placeholder')
    os.chdir('./r_em_params')