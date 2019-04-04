# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:07:15 2019

@author: Thomas Athey
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mpl

"""
Directly compare the EM clustering algorithms of python's GaussianMixture and R's mclust

"""

bic_files = {'full':'VVV','tied':'EEE','diag':'VVI','spherical':'VII'}
agglom_names = ['R-EEE','R-EII','R-VII','R-VVV','Py-average','Py-complete','Py-single','Py-ward']

for py,r in bic_files.items():
    pyfile = 'python_bic_'+py+'.csv'
    rfile = 'r_bic_'+r+'.csv'
    pybic = np.genfromtxt(pyfile,skip_header=1,delimiter=',')
    rbic = np.genfromtxt(rfile,skip_header=1,delimiter=',')
    plt.figure(figsize=(8,8))
    ks = np.arange(pybic.shape[1])
    for i in np.arange(pybic.shape[0]):
        diff = pybic[i,:] - rbic[i,:]
        ok = [all(tup) for tup in zip(pybic[i,:],rbic[i,:])] 
        plt.plot(ks[ok],diff[ok],label=agglom_names[i])
    plt.legend()
    plt.title(py)
    plt.xlabel('k')
    plt.ylabel('pyBIC - RBIC')
    #plt.savefig(py+'.jpg')