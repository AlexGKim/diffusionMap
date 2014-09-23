#!/usr/bin/env python

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def diffuse(array, **kwargs):
    
    dM=importr('diffusionMap')
    nr, nc = array.shape

    xvec = robjects.FloatVector(array.reshape(array.size))
    xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
    xr=robjects.r.dist(xr)

    dmap = robjects.r.diffuse(xr, **kwargs)

    return dmap
