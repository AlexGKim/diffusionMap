#!/usr/bin/env python

import numpy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
import matplotlib.colors
import diffuse

t=1
epsval=22.84
factor=1.4

array=numpy.loadtxt('../data/trainingPCACoeff.dat')

dmap = diffuse.diffuse(array,t=t, eps_val=epsval*factor)

X=numpy.array(dmap.rx('X')[0])
#eigenmult = numpy.array(dmap.rx('eigenmult')[0])
#print eigenmult
#plt.plot(eigenmult,'.')
#plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ndata = X.shape[0]/10
for i in xrange(ndata):
  ax.scatter(X[i*10:(i+1)*10,0],X[i*10:(i+1)*10,1],X[i*10:(i+1)*10,2],norm=matplotlib.colors.Normalize(0,ndata),c=numpy.zeros(10)+i,cmap=plt.get_cmap('gist_rainbow'))
ax.set_xlabel('X[0]')
ax.set_ylabel('X[1]')
ax.set_zlabel('X[2]')
plt.show()
print vfwfe
g1 = X[:,0]<20
g2 = numpy.logical_and(X[:,0]>=20,X[:,2] > 0)
g3 = numpy.logical_and(X[:,0]>=20,X[:,2]<=0)
print numpy.sum(g1),numpy.sum(g2),numpy.sum(g3)
