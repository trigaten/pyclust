# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:31:31 2019

@author: Thomas Athey
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os


"""
Several of python's options for hierarchical agglomerative clustering,
saving the results to csv files
"""

os.chdir('..')
X = np.genfromtxt('embedded_right.csv',delimiter=',',skip_header=1)
n = X.shape[0]
c_true = np.genfromtxt('classes.csv',skip_header=1)
os.chdir('./compare_bic')

linkages = ['single','complete','average','ward']
kmax = 19
ks = np.arange(1,kmax+1)
agglom = np.zeros((n,kmax))

for l in linkages:
    print(l)
    hdr = ''
    for k in ks:
        print(k)
        aggclust = AgglomerativeClustering(n_clusters=k,linkage=l).fit(X)
        agglom[:,kmax-k] = aggclust.labels_.astype('int')
        hdr = hdr + str(kmax-k+1) + ','
    fname = 'python_hc_' + l + '.csv'
    np.savetxt(fname,agglom,fmt='%d',delimiter=',',header=hdr)
    