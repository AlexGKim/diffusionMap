#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse']

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def dist(d):
    '''This function computes and returns the distance matrix computed by
    using the specified distance measure to compute the distances
    between the rows of a data matrix.
    
    Parameters
    ----------
    d: numpy.ndarray with shape (n_samples, n_features).  n-by-m
        feature matrix for a data set with n points and m features.
    '''
    nr, nc = d.shape

    xvec = robjects.FloatVector(d.reshape(d.size))
    xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
    xr=robjects.r.dist(xr)
    
    return xr
    
    

def diffuse(d, **kwargs):
    '''Uses the pair-wise distance matrix for a data set to compute
    the diffusion map coefficients. Computes the Markov transition
    probability matrix, and its eigenvalues and left and right
    eigenvectors. Returns a `robjects.r.dmap` object.
    
    Parameters:
    -----------
    d: numpy.ndarray with shape (n_samples, n_features).  n-by-m
        feature matrix for a data set with n points and m features.
    
    Returns:
    --------
    rpy2.robjects.r.dmap object containing the diffusion map
        coefficients for each sample.
    '''

    dM=importr('diffusionMap')
    xr = dist(d)
    dmap = robjects.r.diffuse(xr, **kwargs)
    return dmap


