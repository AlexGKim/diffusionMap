#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse', 'nystrom']

#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#import gc
import numpy
import numpy as np
import scipy
from scipy.stats import norm
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

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
#   dM=importr('diffusionMap')
    # R distance
    #xr=numpy.array(robjects.r.dist(d))
    # Python distance
    xr = scipy.spatial.distance.pdist(d, 'euclidean')
    xr_arr=scipy.spatial.distance.squareform(xr)
 
#    dmap = dM.diffuse(xr_arr, **kwargs)
    dmap = diffuse_py(xr_arr, **kwargs)

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
    #dM=importr('diffusionMap')
    
    # Concatenate new data to training data

    data_concat = np.concatenate((orig, d))
    n_samp, n_feat = data_concat.shape
    data_concat = data_concat.reshape(n_samp, 1, n_feat)
#    data_concat_T = data_concat.reshape(1, n_samp, n_feat)

    # Compute one big distance matrix
    xr = scipy.spatial.distance.pdist(data_concat[:,0,:], 'euclidean')
    distmat = scipy.spatial.distance.squareform(xr)    
#    distmat = np.sum((data_concat_T - data_concat)**2, axis=2)
    
    # Slice off the part of the distance matrix that 
    # represents distances from new data to original data
    distmat_slice = distmat[len(orig):, :len(orig)]
    # Compute nystrom and return

    # R version
    #coords = dM.nystrom(dmap, distmat_slice, **kwargs)
    # Python version
    coords = nystrom_py(dmap, distmat_slice, **kwargs)

    return coords

def nystrom_py(dmap, Dnew, **kwargs):

    Nnew, Nold  =  Dnew.shape
#    eigenvals=numpy.array(dmap.rx('eigenvals')[0])
    eigenvals=dmap['eigenvals']
#    X=numpy.array(dmap.rx('X')[0])
    X=dmap['X']
#    sigma=dmap.rx('epsilon')[0]
    sigma=dmap['epsilon']
    if Nold != X.shape[0]:
      print "dimensions don't match"
    Xnew = numpy.exp(-Dnew**2/sigma)
    v = numpy.sum(Xnew,axis=1)

    #some points are far away from the training set
    w = numpy.where(v !=0)
    bw = numpy.where( v==0)
    for i in xrange(Xnew.shape[1]):
      #Xnew[:,i]=Xnew[:,i]/v
      Xnew[w,i]=Xnew[w,i]/v[w]
    dum = numpy.array(X)
    for i in xrange(X.shape[1]):
      dum[:,i]=X[:,i]/eigenvals[i]
    Xnew = numpy.dot(Xnew ,dum)
    Xnew[bw,:]=numpy.nan
    return Xnew

def epsilonCompute(D, p=0.01):

#   D=scipy.spatial.distance.squareform(D)
   n = D.shape[0]
   k = numpy.ceil(p*n)
   k = numpy.maximum(2,k)
   D_sort = numpy.sort(D,0)
   dist_knn = D_sort[k,:]
   epsilon = 2*numpy.median(dist_knn)**2
   return epsilon

def diffuse_py(D,eps_val='default',neigen=None,t=0,maxdim=50,delta=1e-5):
  if eps_val == 'default':
    eps_val = epsilonCompute(D)

  n=D.shape[0]
  K=numpy.exp(-D**2/eps_val)
  v = numpy.sqrt(numpy.sum(K,axis=0))
  
  A= K / numpy.outer(v,v)
  del K
  w=numpy.where(A > delta)
  Asp =  csc_matrix( (A[w],(w[0],w[1])), shape=A.shape )
  del A
  if neigen is None:
    neff = numpy.minimum(maxdim,n)
  else:
    neff =  numpy.minimum(neigen, n)
  neff=neff.item() #convert numpy int to int

  evals, evecs = eigsh(Asp, k=neff, which='LA', ncv=numpy.maximum(30,2*neff) )
  psi = evecs[:,neff-1::-1]/numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)
  phi = evecs[:,neff-1::-1]*numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)

  eigenvals = evals[neff-1::-1]#eigenvalues

  if t<=0:
    lambd = eigenvals[1:]/(1-eigenvals[1:])
    lambd= numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.amax(numpy.where(lam > 0.05))+1
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen+1]
    X = psi[:,1:neigen+1]*lambd[:,0:neigen]
  else:
    lambd= eigenvals[1:]**t
    lambd=numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.amax(numpy.where(lam > 0.05))+1
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen+1]
    X = psi[:,1:neigen+1]*lambd[:,0:neigen]
  y=dict()
  y['X']=X
  y['phi0']=phi[:,0]
  y['eigenvals']=eigenvals[1:]
  y['eigenmult']=lambd[:,:neigen]
  y['psi']=psi
  y['phi']=phi
  y['neigen']=neigen
  y['epsilon']=eps_val
  return y
