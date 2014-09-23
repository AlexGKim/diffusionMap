#!/usr/bin/env python

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def function diffuse(nparray, t=t, eps_val=eps_val):
    
    dM=importr('diffusionMap')
    nr, nc = array.shape

    xvec = robjects.FloatVector(array.reshape(array.size))
    xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
    xr=robjects.r.dist(xr)

    dmap = robjects.r.diffuse(xr,t=t, eps_val=epsval*factor)

    return dmap
