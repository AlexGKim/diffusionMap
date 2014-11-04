#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse', 'nystrom']

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np

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
    nr, nc = d.shape
    xvec = robjects.FloatVector(d.reshape(d.size))
    xr=robjects.r.matrix(xvec,nrow=nr,ncol=nc,byrow=True)
    xr=robjects.r.dist(xr)
    dmap = robjects.r.diffuse(xr, **kwargs)
    return dmap

def nystrom(dmap, orig, d, sigma='default'):
    '''Given the diffusion map coordinates of a training data set,
    estimates the diffusion map coordinates of a new set of data using
    the pairwise distance matrix from the new data to the original
    data.
  
    Parameters
    ----------
  
    dmap: a dmap object from the original data set, computed by diffuse()

    d: numpy.ndarray with shape (n_new_samples, n_features), where
      n_features is the same as the training set to dmap

    orig: feature array with shape (n_samples, n_features) that was
      used to train the original dmap.
  
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
    
    # Concatenate new data to training data

    data_concat = np.concatenate((orig, d))
    n_samp, n_feat = data_concat.shape
    data_concat = data_concat.reshape(n_samp, 1, n_feat)
    data_concat_T = data_concat.reshape(1, n_samp, n_feat)

    
    # Compute one big distance matrix
    
    distmat = np.sum((data_concat_T - data_concat)**2, axis=2)
    
    # Slice off the part of the distance matrix that 
    # represents distances from new data to original data
    distmat_slice = distmat[len(orig):, :len(orig)]
    
    nr, nc = distmat_slice.shape

    # Shove into R
    xvec = robjects.FloatVector(distmat_slice.reshape(distmat_slice.size))
    xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc, byrow=True)
    
    # Compute nystrom and return
    coords = robjects.r.nystrom(dmap, xr, **kwargs)
    return coords
