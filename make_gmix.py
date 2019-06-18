import numpy as np

n = 100
d = 10
w = [0.33,0.33,0.34]

k = len(w)
z = np.zeros([d])
means = [np.copy(z) for i in range(k)]
means[1][0] = 5
means[2][1] = 5
cov = [np.identity(d) for i in w]
x = np.zeros([n,d+1])

thresholds = np.cumsum(w)
for i in np.arange(n):
    u = np.random.rand(1)
    component = np.argmax((u < thresholds))
    x[i,0] = component
    x[i,1:] = np.random.multivariate_normal(means[component],cov[component])
    
np.savetxt('data/highd.csv',x,delimiter=',')