import numpy as np

n = 100
d = 3
w = [0.33,0.33,0.34]
means = [[0,0,0],[0,10,0],[0,0,10]]
cov = [np.identity(3) for i in w]
x = np.zeros([n,d+1])

thresholds = np.cumsum(w)
for i in np.arange(n):
    u = np.random.rand(1)
    component = np.argmax((u < thresholds))
    x[i,0] = component
    x[i,1:] = np.random.multivariate_normal(means[component],cov[component])
    
np.savetxt('data/highd.csv',x,delimiter=',')