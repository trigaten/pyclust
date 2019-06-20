#4 tests that show brute_cluster to be working for each type of covariance matrix (spherical, diag, tied, and full)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib #as plt
matplotlib.use('TkAgg') #necessary code as not to cause an error on my machine (OS)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from brute_cluster import brute_cluster
import pytest

type = "spherical" #spherical, diag, tied, or full
k = 3 #number of different clusters to be created
means = [] #initializing blank list
means.append([130, 49, 350])
means.append([420, 49, 200])
means.append([120, 550, 459])
covs = [] #covariance matrices
vectors = np.zeros([300, 3]) #list of all the points 

affinities = ['none', 'euclidean','manhattan','cosine']
linkages = ['none','ward','complete','average','single']
covariance_types = ['full','tied','diag','spherical']
ks = [i for i  in range(1,7)] #different cluster amounts to be tested

def testSpherical():
    covs.append([[500, 0, 0], [0, 500, 0], [0, 0, 500]])
    covs.append([[760, 0, 0], [0, 760, 0], [0, 0, 760]])
    covs.append([[212, 0, 0], [0, 212, 0], [0, 0, 212]])
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    try:
         assert bft == 'spherical'
         print("is spherical")
    except:
        print("assertion error")
    __plot(vectors, 'actual cov matrix: spherical, brute_force choice: ' + bft)


def testDiag():
    covs.append([[300, 0, 0], [0, 613, 0], [0, 0, 700]])
    covs.append([[450, 0, 0], [0, 610, 0], [0, 0, 320]])
    covs.append([[900, 0, 0], [0, 225, 0], [0, 0, 560]])
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    try:
         assert bft == 'diag'
         print("is diag")
    except:
        print("assertion error")
    __plot(vectors, 'actual cov matrix: diag, brute_force choice: ' + bft)

def testTied():
    covs.append([[450, 225, 200], [300, 670, 430], [120, 500, 995]])
    covs.append([[450, 225, 200], [300, 670, 430], [120, 500, 995]])
    covs.append([[450, 225, 200], [300, 670, 430], [120, 500, 995]])
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    try:
         assert bft == 'tied'
         print("is tied")
    except:
        print("assertion error")
    __plot(vectors, 'actual cov matrix: tied, brute_force choice: ' + bft)

def testFull():
    covs.append([[450, 225, 200], [300, 670, 430], [120, 500, 995]])
    covs.append([[75, 225, 150], [90, 120, 70], [40, 555, 60]])
    covs.append([[600, 195, 20], [80, 100, 65], [95, 120, 330]])
    __doAssignments()
    bft = __getCov()#what cov matrix brute_force chooses
    try:
         assert bft == 'full'
         print("is full")
    except:
        print("assertion error")
    __plot(vectors, 'actual cov matrix: full, brute_force choice: ' + bft)

# assigns points to vectors array
def __doAssignments():
    for i in range(k):
        ax, ay, az = np.random.multivariate_normal(means[i], covs[i], 100).T
        vectors[i * 100: (i+1) * 100, 0] = ax
        vectors[i * 100: (i+1) * 100, 1] = ay
        vectors[i * 100: (i+1) * 100, 2] = az

# returns the covariance matrix predicted by the brute_cluster function
def __getCov():
    best_c_hat, best_combo, best_k = brute_cluster(vectors, affinities, linkages, covariance_types, ks, None, None, 1)
    print(best_combo, best_k)
    return best_combo[2]

# plots the model 
def __plot(vectors, title):
    x, y, z = zip(*vectors)
    fig = plt.figure()
    graph = fig.add_subplot(111, projection='3d')
    graph.scatter(x, y, z)
    graph.set_title(title,fontsize=15,fontweight='normal', y=1.1)
    plt.show()




