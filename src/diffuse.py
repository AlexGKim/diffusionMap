#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse', 'nystrom']

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

def nystrom(dmap, d, sigma='default'):
    '''Given the diffusion map coordinates of a training data set,
    estimates the diffusion map coordinates of a new set of data using
    the pairwise distance matrix from the new data to the original
    data.
  
    Parameters
    ----------
  
    dmap: a ’"dmap"’ object from the original data set, computed by diffuse()

    d: numpy.ndarray with shape (n_samples, n_features), where
      n_features is the same as the training set to dmap
  
    sigma: 'default' or int: A scalar giving the size of the Nystrom
      extension kernel. Default uses the tuning parameter of the
      original diffusion map

    Returns:
    --------
  
    The estimated diffusion coordinates for the new data, a matrix of
      dimensions m by p, where p is the dimensionality of the input
      diffusion map.
    '''
    
    
    kwargs = {}
    if sigma != 'default':
        kwargs['sigma'] = sigma

    dM=importr('diffusionMap')
    xr = dist(d)
    coords = robjects.r.nystrom(dmap, xr, **kwargs)
    return coors
