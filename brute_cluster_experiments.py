import numpy as np
from brute_cluster import brute_cluster
import csv



dataset = 0 #0-drosophila, 1-BC, 2-diabetes, 3-gaussianmix
#None - no figures will be saved, string - files will be saved with that name
savefigs = None 

if dataset == 0:
    #Drosophila
    ks = [i for i in range(1,19)]
    affinities = ['none','euclidean','manhattan','cosine']
    linkages = ['none','ward','complete','average','single']
    covariance_types=['full','tied','diag','spherical']
    x = np.genfromtxt('data/embedded_right.csv',delimiter=',',skip_header=1)
    c_true = np.genfromtxt('data/classes.csv',skip_header=1)
elif dataset==1:
    #Wisconsin Diagnostic Data
    ks = [i for i in range(1,10)]
    affinities = ['none','euclidean','manhattan','cosine']
    linkages = ['none','ward','complete','average','single']
    covariance_types=['full','tied','diag','spherical']
    
    #read mean texture, extreme area, and extreme smoothness
    x = np.genfromtxt('data/wdbc.data',delimiter=',', usecols = (3,25,26),skip_header=0)
    with open('data/wdbc.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        c_true = []
        for row in reader:
            c_true.append(row[1])
            
    c_true = np.asarray([int(c == 'M') for c in c_true])
elif dataset==2:
    #Reaven and Miller Diabetes
    ks = [i for i in range(1,21)]
    affinities = ['none','euclidean','manhattan','cosine']
    linkages = ['none','ward','complete','average','single']
    covariance_types=['full','tied','diag','spherical']
    
    #read glucose area, insulin area, and SSPG
    x = np.genfromtxt('data/T36.1', delimiter=',', usecols = (6,7,8),skip_header=0)
    c_true = np.genfromtxt('data/T36.1', delimiter=',', usecols = (9),skip_header=0)
elif dataset==3:
    ks = [i for i in range(1,21)]
    affinities = ['none','euclidean','manhattan','cosine']
    linkages = ['none','ward','complete','average','single']#,'ward','complete']
    covariance_types=['full','tied','diag','spherical']
    
    x = np.genfromtxt('data/synthetic.csv', delimiter=',',skip_header=0)
    x = x[:,np.arange(1,x.shape[1])]
    c_true = np.genfromtxt('data/synthetic.csv', delimiter=',', usecols = (0),skip_header=0)


c_hat,_ = brute_cluster(x, affinities, linkages, covariance_types, ks,
                           c_true,savefigs)