#4 tests that show brute_cluster to be working for each type of covariance matrix (spherical, diag, tied, and full)
import sys
sys.path.append("C:\\Users\\Thomas Athey\\Documents\\Labs\\Labs\\jovo\\clustering\\pyclust")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib #as plt
matplotlib.use('TkAgg') #necessary code as not to cause an error on my machine (OS)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from brute_cluster import brute_cluster

import pytest

k = 3 #number of different clusters to be created
means = [] #initializing blank list
means.append([500, 200])
means.append([200, 600])
means.append([100, 100])
covs = [0, 0, 0] #covariance matrices
vectors = np.zeros([600, 2]) #list of all the points 

affinities = ['none', 'euclidean','manhattan','cosine']
linkages = ['none','ward','complete','average','single']
covariance_types = ['full','tied','diag','spherical']
ks = [i for i  in range(1,7)] #different cluster amounts to be tested

def testSpherical():
    covs[0] = [[500, 0], [0, 500]]
    covs[1] = [[700, 0], [0, 700]]
    covs[2] = [[200, 0], [0, 200]]
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    __plot(vectors, 'actual cov matrix: spherical, brute_force choice: ' + bft)
    assert bft == 'spherical'

def testDiag():
    covs[0] = [[100, 0], [0, 1300]]
    covs[1] = [[1000, 0], [0, 100]]
    covs[2] = [[1200, 0], [0, 100]]
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    assert bft == 'diag'
    __plot(vectors, 'actual cov matrix: diag, brute_force choice: ' + bft)

def testTied():
    covs[0] = [[100, 500], [300, 900]]
    covs[1] = [[100, 500], [300, 900]]
    covs[2] = [[100, 500], [300, 900]]
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    assert bft == 'tied'
    __plot(vectors, 'actual cov matrix: tied, brute_force choice: ' + bft)

def testFull():
    covs[0] = [[400, 900], [900, 600]]
    covs[1] = [[900, 100], [700, 400]]
    covs[2] = [[600, 1000], [200, 300]]
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    assert bft == 'full'
    __plot(vectors, 'actual cov matrix: full, brute_force choice: ' + bft)

# assigns points to vectors array
def __doAssignments():
    for i in range(k):
        ax, ay = np.random.multivariate_normal(means[i], covs[i], 200).T
        vectors[i * 200: (i+1) * 200, 0] = ax
        vectors[i * 200: (i+1) * 200, 1] = ay

# returns the covariance matrix predicted by the brute_cluster function
def __getCov():
    best_c_hat, best_combo, best_k = brute_cluster(vectors, affinities, linkages, covariance_types, ks, None, None, 1)
    print(best_combo, best_k)
    return best_combo[2]

# plots the model 
def __plot(vectors, title):
    x, y = zip(*vectors)
    fig = plt.figure()
    graph = fig.add_subplot(111)
    graph.scatter(x, y)
    graph.set_title(title,fontsize=15,fontweight='normal', y=1.0)
    plt.show()
