#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse', 'nystrom']

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy
import numpy as np
import gc

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
    xr=robjects.r.dist(d)

    # must be better way to do this but convert returned lower triangualr
    # array into an array
    xr_arr=np.zeros((d.shape[0],d.shape[0]))
    for i in xrange(0,d.shape[0]-1):
      for j in xrange(i+1,d.shape[0]):
        xr_arr[i,j]=xr[d.shape[0]*i - i*(i+1)/2 + j-i -1]
        xr_arr[j,i]=xr_arr[i,j]
    dmap = dM.diffuse(xr_arr, **kwargs)
    
    # garbage collection
    gc.collect()

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

    # Compute nystrom and return
    coords = dM.nystrom(dmap, distmat_slice, **kwargs)

    # python garbage collection
    gc.collect()
    return coords
