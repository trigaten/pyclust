# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:19:55 2019

These functions allow for "brute clustering," inspired by R's mclust.

Clustering is performed first by hierarchical agglomeration, then fitting a
Gaussian Mixture via Expectation Maximization (EM). There are several ways to 
perform both agglomeration and EM so these functions performs the (specified)
combinations of methods then evaluates each according to BIC.

@author: Thomas Athey
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from compare_bic.bic import processBIC
from scipy import stats

colors = ['red','green','blue','orange','purple','yellow','black','brown',
          'lightsalmon','greenyellow','cornflowerblue','tan','violet',
          'gold','slategray','peru','indianred','darkolivegreen',
          'navy','darkgoldenrod','deeppink','darkkhaki','silver','saddlebrown']


def agglomerate(data, aff, link, k):
    """
    Hierarchical Agglomeration
    inputs:
        data - nxd numpy array
        aff - affinity technique, an element of ['euclidean','manhattan','cosine']
        link - linkage technique, an element of ['ward','complete','average','single']
        k - number of clusters
    outputs:
        one_hot - nxk numpy array with a single one in each row indicating cluster
            membership
    exceptions:
        ward linkage can only be used with euclidean/l2 affinity so if ward is 
        specified with a different linkage then there is an Exception
    """
    n=data.shape[0]
    if link=='ward' and aff !='euclidean': 
        raise Exception('Ward linkage is only valid with Euclidean affinity')
    agglom = AgglomerativeClustering(n_clusters=k,affinity=aff,linkage=link).fit(data)
    one_hot = np.zeros([n,k])
    one_hot[np.arange(n),agglom.labels_] = 1
    return one_hot


def initialize_params(data, one_hot, cov):
    """
    sklearn's Gaussian Mixture does not allow initialization from class membership
    but it does allow from initialization of mixture parameters, so here we calculate
    the mixture parameters according to class membership

    input:
        data - nxd numpy array 
        one_hot - nxd numpy array with a single one in each row indicating cluster
            membership
        k - number of clusters
    output:
        weights - k array of mixing weights
        means - kxd array of means of mixture components
        precisions - precision matrices, format depends on the EM clustering option
            (eg 'full' mode needs a list of matrices, one for each mixture
            component,but 'tied' mode only needs a single matrix, since all
            precisions are constrained to be equal)
    """
    n=data.shape[0]
    weights, means, covariances = _estimate_gaussian_parameters(
            data, one_hot, 1e-06, cov)
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
    
    return weights, means, precisions
    
    


def cluster(data, aff, link, cov, k, c_true=None):
    """
    Cluster according to specified method

    input:
        data - nxk numpy matrix of data
        c_true - n array of true cluster membership
        aff - affinity, element of ['euclidean','manhattan','cosine'] or none for EM from scratch
        link - linkage, element of ['ward','complete','average','single'], or none for EM from scratch
        cov - covariance, element of ['full','tied','diag','spherical']
        k - # of clusters
    output:
        c_hat - n array of clustering results
        means - kxd array of means of mixture components
        bic - Bayes Information Criterion for this clustering
        ari - Adjusted Rand Index to comparing clustering result to true clustering
        reg - regularization parameter that was used in the clustering results
            (0 or 1e-6)
    """
    means = None
    if aff=='none' or link=='none':
        reg = 1e-6
        while True:
            try:
                gmm = GaussianMixture(
                n_components=k,covariance_type=cov,reg_covar=reg)
                c_hat = gmm.fit_predict(data)
                bic = -gmm.bic(data) #processBIC(data,gmm.weights_,gmm.means_,gmm.covariances_,cov)
                means = gmm.means_
                if any([sum(c_hat == i)<=1 for i in range(k)]) or bic==-np.inf:
                    raise ValueError
                else:
                    break
            except:
                pass
            if reg == 0:
                reg = 1e-6
            elif reg > 1:
                bic = -np.inf
                break
            else:
                reg = reg*10
    else:
        one_hot = agglomerate(data,aff,link,k)
        init_weights, init_means, init_precisions = initialize_params(data, one_hot, cov)
        
        reg = 0
        while True:
            try:
                gmm = GaussianMixture(
                        n_components=k,covariance_type=cov,weights_init=init_weights,
                        means_init=init_means,precisions_init=init_precisions)
                c_hat = gmm.fit_predict(data)
                bic = -gmm.bic(data) #processBIC(data,gmm.weights_,gmm.means_,gmm.covariances_,cov)
                means = gmm.means_
                if any([sum(c_hat == i)<=1 for i in range(k)]) or bic==-np.inf:
                    raise ValueError
                else:
                    break
            except:
                pass
            if reg == 0:
                reg = 1e-6
            elif reg > 1:
                bic = -np.inf
                break
            else:
                reg = reg*10
    
    if c_true is not None:    
        ari = adjusted_rand_score(c_true,c_hat)
    else:
        ari = None
        
    
    return c_hat, means, bic, ari, reg


def brute_cluster(x, affinities, linkages,
                  covariance_types, ks, c_true=None, savefigs=None, verbose=0, graphList=None):
    """
    Cluster all combinations of options and plot results
    inputs:
        x - nxd array of data
        affinites - list of affinity modes, each must be an element of
            ['none,'euclidean','manhattan','cosine']
        linkages - list of linkage modes, each must be an element of
            ['none','ward','complete','average','single']
        covariance_types - list of covariance modes, each must be an element of
            ['full','tied','diag','spherical']
        ks - list of cluster numbers
        c_true - n array of true clustering
        savefigs - None indicates that figures should not be saved, a string value
            indicates the name that should be used when saving the figures
        verbose - if 0, no output, if 1, output the current clustering options
            being used
        graphList - if None, all graphs displayed. Is a 1d list of strings each corresponding to the type of graph that can be output
            ['best_bic','true','best_ari','ari_vs_bic','all_bics']
            If, for example, ['ari', 'bic'] is input, the ari and bic graphs will be displayed
    outputs:
        best_c_hat_bic - nx1 array of cluster membership
        best_combo_bic - list of three strings identifying the affinity, linkage, and covariance type that gave best BIC
            (see earlier comments for possibilities)
        best_k_bic - integer indicating the number of clusters that gave best BIC
    """
    cov_dict = {'full':0, 'tied':1,'diag':2,'spherical':3}
    aff_dict = {'none':0,'euclidean':0,'manhattan':1,'cosine':2}
    link_dict = {'none':0,'ward':1, 'complete':2, 'average':3, 'single':4}
    #11 agglomeration combos: 4 with l2 affinity, 3 with l1, 3 with cos, and no agglom
    #4 EM options: full, tied, diag, spherical
    bics = np.zeros([44,len(ks)]) - np.inf
    aris = np.zeros([44,len(ks)]) - np.inf
    
    best_ari = float('-inf')
    best_bic = float('-inf')
    
    for i,k in enumerate(ks):
        for af in affinities:
            for li in linkages:
                if li=='ward' and af !='euclidean':
                    continue
                if (li=='none' and af != 'none') or (af=='none' and li != 'none'):
                    continue
                for cov in covariance_types:
                    if verbose == 1:
                        print('K='+str(k)+' Affinity= '+af+' Linkage= '+li+
                              ' Covariance= ' + cov)
                    row = 11*cov_dict[cov] + 3*aff_dict[af] + link_dict[li]
                    
                    c_hat, means, bic, ari, reg = cluster(x, af,li,cov,
                                                          k, c_true)
                
                    bics[row,i] = bic
                    aris[row,i] = ari
                    
                    if c_true is not None and ari > best_ari:
                        best_ari = ari
                        best_combo_ari = [af,li,cov]
                        best_c_hat_ari = c_hat
                        best_k_ari = k
                        
                    if bic > best_bic:
                        best_bic = bic
                        best_combo_bic = [af,li,cov]
                        best_c_hat_bic = c_hat
                        best_k_bic = k
                        best_ari_bic = ari
                        best_means_bic = means
                        reg_bic = reg
    


    titles = ['full','tied','diag','spherical']
    if graphList == None or 'best_bic' in graphList:
        #Plot with best BIC*********************************
        if c_true is None:
            best_ari_bic_str = 'NA'
        else:
            best_ari_bic_str = '%1.3f'%best_ari_bic

        fig_bestbic = plt.figure(figsize=(8,8))
        ax_bestbic = fig_bestbic.add_subplot(1,1,1)
        #ptcolors = [colors[i] for i in best_c_hat_bic]
        ax_bestbic.scatter(x[:,0],x[:,1],c=best_c_hat_bic)
        #mncolors = [colors[i] for i in np.arange(best_k_bic)]
        mncolors = [i for i in np.arange(best_k_bic)]
        ax_bestbic.scatter(best_means_bic[:,0],best_means_bic[:,1],
                    c=mncolors,marker='x')
        ax_bestbic.set_title("py(agg-gmm) BIC %3.0f from "%best_bic +
                str(best_combo_bic) + " k=" + str(best_k_bic) +
                ' ari=' + best_ari_bic_str + ' reg=' + str(reg_bic))# + "iter=" + str(best_iter_bic))
        ax_bestbic.set_xlabel("First feature")
        ax_bestbic.set_ylabel("Second feature")
        if savefigs is not None:
            plt.savefig(savefigs+'_python_bestbic.jpg')

        
    if c_true is not None:
        if graphList == None or 'true' in graphList:
            #True plot**********************************
            fig_true = plt.figure(figsize=(8,8))
            ptcolors = [colors[i] for i in c_true.astype(int)]
            ax_true = fig_true.add_subplot(1,1,1)
            ax_true.scatter(x[:,0],x[:,1],c=ptcolors)
            ax_true.set_title("True labels")
            ax_true.set_xlabel("First feature")
            ax_true.set_ylabel("Second feature")
            fig_true.show()
            if savefigs is not None:
                plt.savefig(savefigs+'_python_true.jpg')

        if graphList == None or 'best_ari' in graphList:
            #Plot with best ARI************************************
            fig_bestari = plt.figure(figsize=(8,8))
            ax_bestari = fig_bestari.add_subplot(1,1,1)
            ptcolors = [colors[i] for i in best_c_hat_ari]
            ax_bestari.scatter(x[:,0],x[:,1],c=ptcolors)
            ax_bestari.set_title("py(agg-gmm) ARI %3.3f from "%best_ari +
                    str(best_combo_ari) + " k=" + str(best_k_ari))# + "iter=" + str(best_iter_ari))
            ax_bestari.set_xlabel("First feature")
            ax_bestari.set_ylabel("Second feature")
            if savefigs is not None:
                plt.savefig(savefigs+'_python_bestari.jpg')
        
        if graphList == None or 'ari_vs_bic' in graphList:
            #ARI vs BIC********************************
            fig_bicari = plt.figure(figsize=(8,8))
            ax_bicari = fig_bicari.add_subplot(1,1,1)
            for row in np.arange(4):
                xs = bics[row*11:(row+1)*11,:]
                ys = aris[row*11:(row+1)*11,:]
                idxs = (xs!=-np.inf)*(ys!=-np.inf)
                ax_bicari.scatter(xs[idxs],ys[idxs],
                            label=titles[row])
                
            idxs = (bics!=-np.inf)*(aris!=-np.inf)
            _,_,r_value,_,_ = stats.linregress(bics[idxs],aris[idxs])
            ax_bicari.set_xlabel("BIC")
            ax_bicari.set_ylabel("ARI")
            ax_bicari.legend(loc='best',title='Agglomeration Method')
            ax_bicari.set_title("Pyclust's ARI vs BIC for Drosophila Data with Correlation r^2=%2.2f"%(r_value**2))
            if savefigs is not None:
                plt.savefig(savefigs+'_python_bicari.jpg')

    if graphList == None or 'all_bics' in graphList:
        #plot of all BICS*******************************
        labels = {0:'none',1:'l2/ward',2:'l2/complete',3:'l2/average',4:'l2/single',
                5:'l1/complete',6:'l1/average',7:'l1/single',8:'cos/complete',
                9:'cos/average',10:'cos/single'}
        
        _, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharey='row',sharex='col',figsize=(10,10))
        for row in np.arange(bics.shape[0]):
            if all(bics[row,:]==-np.inf):
                continue
            if row<=10:
                ax0.plot(np.arange(1,len(ks)+1),bics[row,:])
            elif row<=21:
                ax1.plot(np.arange(1,len(ks)+1),bics[row,:],label=labels[row%11])
            elif row<=32:
                ax2.plot(np.arange(1,len(ks)+1),bics[row,:])
            elif row<=43:
                ax3.plot(np.arange(1,len(ks)+1),bics[row,:])
        
        ax0.set_title(titles[0],fontsize=20,fontweight='bold')
        ax0.set_ylabel('BIC',fontsize=20)
        ax0.locator_params(axis='y',tight=True,nbins=4)
        ax0.set_yticklabels(ax0.get_yticks(),fontsize=14)

        ax1.set_title(titles[1],fontsize=20,fontweight='bold')
        legend = ax1.legend(loc='best',title='Agglomeration\nMethod',fontsize=12)
        plt.setp(legend.get_title(),fontsize=14)
        ax1.set_yticklabels(ax1.get_yticks(),fontsize=14)

        ax2.set_title(titles[2],fontsize=20,fontweight='bold')
        ax2.set_xlabel('Number of components',fontsize=20)
        ax2.set_xticklabels(ax2.get_xticks(),fontsize=14)
        ax2.set_ylabel('BIC',fontsize=20)
        ax2.locator_params(axis='y',tight=True,nbins=4)
        ax2.set_yticklabels(ax2.get_yticks(),fontsize=14)
        

        ax3.set_title(titles[3],fontsize=20,fontweight='bold')
        ax3.set_xlabel('Number of components',fontsize=20)
        ax3.set_xticklabels(ax3.get_xticks(),fontsize=14)

        if savefigs is not None:
            plt.savefig(savefigs+'_python_bicplot.jpg')

        plt.show()
    return best_c_hat_bic, best_combo_bic, best_k_bic
    

