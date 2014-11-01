#!/usr/bin/env python

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def diffuse(d, **kwargs):
    '''Uses the pair-wise distance matrix for a data set to compute
    the diffusion map coefficients. Computes teh Markov transition
    probability matrix, and its eigenvalues and left and reight
    eigenvectors. Returns a `robjects.r.dmap` object.
    
    Parameters:
    -----------
    d: numpy.ndarray with shape (n_samples, n_samples).  n-by-n
        pairwise distance matrix for a data set with n points or
        alternatively the output from the rpy2.robjects.r.dist
        function.
    
    Returns:
    --------
    rpy2.robjects.r.dmap object containing the diffusion map
        coefficients for each sample.
    '''

    
    dM=importr('diffusionMap')
    nr, nc = array.shape

    xvec = robjects.FloatVector(array.reshape(array.size))
    xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
    xr=robjects.r.dist(xr)

    dmap = robjects.r.diffuse(xr, **kwargs)

    return dmap
