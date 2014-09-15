#!/usr/bin/env python

import numpy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt


dM=importr('diffusionMap')

array=numpy.loadtxt('../data/trainingPCACoeff.dat')
array = array[:,0:25]
nr, nc = array.shape

xvec = robjects.FloatVector(array.reshape(array.size))
xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
xr=robjects.r.dist(xr)

dmap = robjects.r.diffuse(xr)
X=numpy.array(dmap.rx('X')[0])

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()
