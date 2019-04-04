# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:19:55 2019

This script clusters the drosophila data. It searches over all possible
affinities, linkages, and covariance types, and k for the best ari and bic.
I also made it search through all iteration numbers from 1 to 100. 
The best clusterings are shown as plots and the results of all combinations is
saved in a text document.

Findings:
    -see show_clusters pdf

@author: Thomas Athey
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import os,sys
from compare_bic.bic import processBIC

colors = ['red','green','blue','orange','purple','yellow','black','brown',
          'lightsalmon','greenyellow','cornflowerblue','tan','violet',
          'gold','slategray','peru','indianred','darkolivegreen',
          'navy','darkgoldenrod','deeppink','darkkhaki','silver','saddlebrown']

affinities = ['euclidean','manhattan','cosine']
linkages = ['ward','complete','average','single']
covariance_types=['full','tied','diag','spherical']

"""
Brute searches over python's clustering methods to find the best methods 
according to BIC and ARI

The models it searches include GMM from scratch, then combinations of
hierarchical agglomeration followed by GMM

Inputs:
    -ks - list of number of clusters
    -bicplot - boolean, true will generate plots of BIC vs k for each of the clustering options
    -name - string that will be used in the filenames it saves
"""
def best_bicari(ks, bicplot, name):
    #ks, boolean about whether to make bicplot, name of files
    x = np.genfromtxt('embedded_right.csv',delimiter=',',skip_header=1)
    n = x.shape[0]
    c_true = np.genfromtxt('classes.csv',skip_header=1)
    best_ari = 0;
    best_bic = float('inf')
    bics = np.zeros([40,len(ks)])
    
    #First we look at GMM without any sort of agglomerative initialization
    bics_noagglom = np.zeros([4,len(ks)])
    for i,k in enumerate(ks):
        for row,cov in enumerate(covariance_types):
            try:
                gmm = GaussianMixture(
                        n_components=k,covariance_type=cov,reg_covar=0,
                        verbose=0,verbose_interval=1, warm_start=True)
                typ=1
                c_hat = gmm.fit_predict(x)
                if any([sum(c_hat == i)<=1 for i in range(k)]):
                    gmm.set_params(reg_covar=1e-6)
                    typ = 2
                    c_hat = gmm.fit_predict(x)
            except ValueError:
                gmm = GaussianMixture(
                        n_components=k,covariance_type=cov,reg_covar = 1e-6,
                        verbose=0,verbose_interval=1)
                typ=3
                c_hat = gmm.fit_predict(x)
            try:
                bic = -processBIC(x,gmm.weights_,gmm.means_,
                                 gmm.covariances_,cov)
            except np.linalg.LinAlgError:
                bic = np.Inf
                
            ari = adjusted_rand_score(c_true,c_hat)
            bics_noagglom[row,i] = bic
            
            if ari > best_ari:
                best_ari = ari
                best_combo_ari = ['none','none',cov]
                best_c_hat_ari = c_hat
                best_k_ari = k
                #best_iter_ari = iter_num
            
            if bic < best_bic:
                best_bic = bic
                best_combo_bic = ['none','none',cov]
                best_c_hat_bic = c_hat
                #print(c_hat)
                best_k_bic = k
                best_ari_bic = ari
                best_means_bic = gmm.means_
                reg_bic = typ
                #best_iter_bic = iter_num
    
    #Now we look at agglomerative initializations
    for i,k in enumerate(ks):
        row = -1
        
        for af in affinities:
            #print('***********************************')
            #print(af)
            for li in linkages:
                if li=='ward' and af !='euclidean':
                    continue
                #print('--------------------------')
                #print(li)
                agglom = AgglomerativeClustering(n_clusters=k,affinity=af,linkage=li).fit(x)
                one_hot = np.zeros([n,k])
                one_hot[np.arange(n),agglom.labels_] = 1
                for cov in covariance_types:
                    row+=1
                    #to initialize GMM, we must calculate the initial gaussian
                    #parameters from the cluster assignments
                    weights, means, covariances = _estimate_gaussian_parameters(
                            x, one_hot, 1e-06, cov)
                    weights /= n
                    
                    precisions_cholesky_ = _compute_precision_cholesky(
                        covariances, cov)
                    if cov=='tied':
                        c = precisions_cholesky_
                        precisions = np.dot(c,c.T)
                    elif cov=='diag':
                        precisions = precisions_cholesky_
                    else:
                        precisions = [np.dot(c,c.T) for c in precisions_cholesky_]
                    #for iter_num in np.arange(1,101): #if we were going to iterate over iteration numbers
                    iter_num=100
                    #print('k='+str(k) + ', affinity=' + af + ', linkage=' + li + ', cov='+cov + ', iter=' + str(iter_num))
                    try: #no regularization
                        gmm = GaussianMixture(
                                n_components=k,covariance_type=cov,weights_init=weights,
                                means_init=means,precisions_init=precisions,
                                max_iter=iter_num, reg_covar=0,
                                verbose=0,verbose_interval=1, warm_start=True)
                        typ=1 
                        c_hat = gmm.fit_predict(x)
                        #if there are any clusters with a single datapoint - regularize 
                        if any([sum(c_hat == i)<=1 for i in range(k)]):
                            gmm.set_params(reg_covar=1e-06)
                            typ = 2
                            c_hat = gmm.fit_predict(x)
                    except ValueError: #if there was a numerical error - regularize
                        gmm = GaussianMixture(
                                n_components=k,covariance_type=cov,weights_init=weights,
                                means_init=means,precisions_init=precisions,
                                max_iter=iter_num, reg_covar = 1e-6,
                                verbose=0,verbose_interval=1)
                        typ=3
                        
                        c_hat = gmm.fit_predict(x)
                    ari = adjusted_rand_score(c_true,c_hat)
                    try:
                        bic = -processBIC(x,gmm.weights_,gmm.means_,
                                         gmm.covariances_,cov)
                    except np.linalg.LinAlgError: #if we have invalid covariances - ignore
                        bic = np.Inf
                    bics[row,i] = bic
                    #print(bic)
                    
                    if ari > best_ari:
                        best_ari = ari
                        best_combo_ari = [af,li,cov]
                        best_c_hat_ari = c_hat
                        best_k_ari = k
                        #best_iter_ari = iter_num
                        
                    if bic < best_bic:
                        best_bic = bic
                        best_combo_bic = [af,li,cov]
                        best_c_hat_bic = c_hat
                        #print(c_hat)
                        best_k_bic = k
                        best_ari_bic = ari
                        best_means_bic = gmm.means_
                        reg_bic = typ
                        #best_iter_bic = iter_num
                        
                    #print(cov)
                    #print(ari)
    
    #True plot
    plt.figure(figsize=(8,8))
    ptcolors = [colors[i] for i in c_true.astype(int)]
    plt.scatter(x[:,0],x[:,1],c=ptcolors)
    plt.title("True labels")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    #plt.savefig('true.jpg')
    
    #BIC Plot
    plt.figure(figsize=(8,8))
    #ptcolors = [colors[i] for i in best_c_hat_bic]
    plt.scatter(x[:,0],x[:,1],c=best_c_hat_bic)
    #mncolors = [colors[i] for i in np.arange(best_k_bic)]
    mncolors = [i for i in np.arange(best_k_bic)]
    plt.scatter(best_means_bic[:,0],best_means_bic[:,1],
                c=mncolors,marker='x')
    plt.title("py(agg-gmm) BIC %3.0f from "%best_bic +
              str(best_combo_bic) + " k=" + str(best_k_bic) +
              ' ari=%1.3f'%best_ari_bic + ' reg=' + str(reg_bic))# + "iter=" + str(best_iter_bic))
    plt.legend()
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    #plt.savefig('python_bic_' + name + '.jpg')
    
    #ARI plot
    plt.figure(figsize=(8,8))
    ptcolors = [colors[i] for i in best_c_hat_ari]
    plt.scatter(x[:,0],x[:,1],c=ptcolors)
    plt.title("py(agg-gmm) ARI %3.3f from "%best_ari +
              str(best_combo_ari) + " k=" + str(best_k_ari))# + "iter=" + str(best_iter_ari))
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    #plt.savefig('python_ari_' + name + '.jpg')

    #print BIC vs k for each model type
    if bicplot:
        ldg = ['l2/ward','l2/complete','l2/average','l2/single','l1/complete','l1/average','l1/single','cos/complete','cos/average','cos/single']
        f, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharey='row',sharex='col',figsize=(10,10))
        for row in np.arange(bics.shape[0]):
            if row%4==0:
                ax0.plot(np.arange(1,len(ks)+1),bics[row,:])
            elif row%4==1:
                ax1.plot(np.arange(1,len(ks)+1),bics[row,:],label=ldg[row//4])
            elif row%4==2:
                ax2.plot(np.arange(1,len(ks)+1),bics[row,:])
            elif row%4==3:
                ax3.plot(np.arange(1,len(ks)+1),bics[row,:])
        ax0.plot(np.arange(1,len(ks)+1),bics_noagglom[0,:])
        ax1.plot(np.arange(1,len(ks)+1),bics_noagglom[1,:],label='no agglom')
        ax2.plot(np.arange(1,len(ks)+1),bics_noagglom[2,:])
        ax3.plot(np.arange(1,len(ks)+1),bics_noagglom[3,:])
        
        ax0.set_title('full')
        ax0.set(ylabel='bic')
        ax1.set_title('tied')
        ax1.legend(loc='lower right')
        ax2.set_title('diag')
        ax2.set(xlabel='k')
        ax2.set(ylabel='bic')
        ax3.set_title('spherical')
        ax3.set(xlabel='k')
        #plt.savefig('python_bicplot.jpg')

ks = [i for i in range(1,26)]
best_bicari(ks,True,'allk')
