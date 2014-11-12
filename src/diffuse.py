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

def nystrom_(dmap, orig, d, sigma='default'):
    kwargs = {}
    if sigma != 'default':
        kwargs['sigma'] = sigma

    # Concatenate new data to training data

    data_concat = np.concatenate((orig, d))
    n_samp, n_feat = data_concat.shape
    data_concat = data_concat.reshape(n_samp, 1, n_feat)
    data_concat_T = data_concat.reshape(1, n_samp, n_feat)


    # Compute one big distance matrix

    distmat = np.sum((data_concat_T - data_concat)**2, axis=2)

    # Slice off the part of the distance matrix that 
    # represents distances from new data to original data
    Dnew = distmat[len(orig):, :len(orig)]
    # Compute nystrom and return
    #coords = dM.nystrom(dmap, distmat_slice, **kwargs)
 

    Nnew, Nold  =  Dnew.shape
    #print Nnew, Nold
    eigenvals=numpy.array(dmap.rx('eigenvals')[0])
    #print 'eigenvals',eigenvals, numpy.amin(eigenvals),numpy.amax(eigenvals)
    X=numpy.array(dmap.rx('X')[0])
#    print X[1,:]
#    few
#    print X.shape (ndata, ndim)
    sigma=dmap.rx('epsilon')[0]

    if Nold != X.shape[0]:
      print "dimensions don't match"

    Xnew = numpy.exp(-Dnew**2/sigma)
#    print 'Xnew',Xnew.shape,numpy.amin(Xnew), numpy.amax(Xnew)
#    rXnew=numpy.array(Xnew)
    v = numpy.sum(Xnew,axis=1)
#    import sys
#    w=numpy.array(numpy.where(v > 10*sys.float_info.epsilon))
#    w=w[0,:]
#    print numpy.amin(v), numpy.amax(v)
    for i in xrange(Xnew.shape[1]):
      Xnew[:,i]=Xnew[:,i]/v
#    print 'Xnew',Xnew.shape,numpy.amin(Xnew), numpy.amax(Xnew)
#    Xnew_w= Xnew[w,:]
#    print 'Xnew_w', Xnew_w.shape,Xnew_w
#    emat=numpy.zeros((eigenvals.shape[0],eigenvals.shape[0]))
#    numpy.fill_diagonal(emat,1/eigenvals)
#    dum = numpy.dot(X,emat)
    dum = numpy.array(X)
    for i in xrange(X.shape[1]):
#      print 1/eigenvals[i]
#      print X[:,i]
      dum[:,i]=X[:,i]/eigenvals[i]
#    print 'first dot',dum.shape, dum, numpy.amin(dum), numpy.amax(dum)
#    Xnew = numpy.dot(Xnew ,numpy.dot( X , emat))
    Xnew = numpy.dot(Xnew ,dum)
#    print 'Xnew', Xnew.shape, Xnew
#    Xnew_w = numpy.dot(Xnew_w ,numpy.dot( X , emat))
#    print 'Xnew_w',Xnew_w.shape, Xnew_w
#    Xnew = numpy.empty((Nnew,X.shape[1]))
#    Xnew[:,:]=numpy.NaN
#    Xnew[w,:]=Xnew_w
#    Xnew = numpy.dot(Xnew ,numpy.dot( X , emat))

#  Xnew = exp(-Dnew^2/(sigma))
  #  v = apply(Xnew, 1, sum)
#  Xnew = Xnew/matrix(v,Nnew ,Nold)
#  #nystrom extension:
#  Xnew = Xnew %*% dmap$X %*% diag(1/dmap$eigenvals)
#    print 'Xnew',Xnew
#    few
    return Xnew


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
#    print distmat_slice.shape, np.amax(distmat_slice), np.amin(distmat_slice)
#    from rpy2.robjects.numpy2ri import numpy2ri
#    rdm = numpy2ri(distmat_slice)
#    rdmap = numpy2ri(dmap)
#    robjects.r.assign('dmap',rdmap)
#    robjects.r.assign('distmat',rdm)
#    robjects.r.save('dmap','distmat',file='map.rsave')
#    for i in xrange(0,distmat.shape[0]-1):
#      for j in xrange(i+1,distmat.shape[1]):
#        print distmat_slice[i,j], distmat_slice[j,i]
#    print dmap.rx('eigenvals')[0], dmap.rx('epsilon')[0]
    # Compute nystrom and return
    coords = dM.nystrom(dmap, distmat_slice, **kwargs)
#    print type(coords), coords.shape, np.min(coords),np.max(coords)
    # python garbage collection
    gc.collect()
    return coords
