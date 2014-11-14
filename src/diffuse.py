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
import scipy
from scipy.stats import norm
from scipy.sparse import *

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

    # R distance
    #xr=numpy.array(robjects.r.dist(d))
    # Python distance
    xr = scipy.spatial.distance.pdist(d, 'euclidean')
#    file = open('dist.txt','w')
#    for x in xr:
#      file.write(str(x)+'\n')
#    file.close()
#    print d.shape, xr.shape
#    fwf
#    print 'distance',xr.min(),xr.max()
    xr_arr=scipy.spatial.distance.squareform(xr)
    #kwargs['eps.val']=2*numpy.sort(xr)[10]**2 #dM.epsilonCompute(xr_arr,p=5e-3)[0]
#    kwargs['eps_val']=dM.epsilonCompute(xr_arr,p=5e-3)[0]
#    kwargs['neigen']=10
#    kwargs['maxdim']=100
#    kwargs['t']=0
#    print kwargs
 
#    dmap = dM.diffuse(xr_arr, **kwargs)
#    print 'R', numpy.amin(dmap.rx('eigenvals')[0]), numpy.amax(dmap.rx('eigenvals')[0])
#    print 'R', numpy.amin(dmap.rx('psi')[0]), numpy.amax(dmap.rx('psi')[0])
#    print 'R', numpy.amin(dmap.rx('phi')[0]), numpy.amax(dmap.rx('phi')[0])
#    print 'R', numpy.amin(dmap.rx('X')[0]), numpy.amax(dmap.rx('X')[0])
    dmap = diffuse_py(xr_arr, **kwargs)
    print dmap['X'].min(), dmap['X'].min()
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

    # R version
    coords = dM.nystrom(dmap, distmat_slice, **kwargs)
    # Python version
    #coords = nystrom_py(dmap, distmat_slice, **kwargs)

    # python garbage collection
    gc.collect()
    return coords

def nystrom_py(dmap, Dnew, **kwargs):

    Nnew, Nold  =  Dnew.shape
    eigenvals=numpy.array(dmap.rx('eigenvals')[0])
    X=numpy.array(dmap.rx('X')[0])
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
#epsilonCompute <- function(D,p=.01){

#  D = as.matrix(D)
#  n = dim(D)[1]
#  k = ceiling(p*n)
#  k = ifelse(k<2,2,k) # use k of at least 2
#  D.sort = apply(D,1,sort)
#  dist.knn = D.sort[(k+1),] # find dists. to kth nearest neighbor
#  epsilon = 2*median(dist.knn)^2
#
#  return(epsilon)
#}


#diffuse <- function(D,eps.val=epsilonCompute(D),neigen=NULL,t=0,maxdim=50,delta=10^-5) {
def diffuse_py(D,eps_val='default',neigen=None,t=0,maxdim=50,delta=1e-5):
  if eps_val == 'default':
    eps_val = epsilonCompute(D)

  n=D.shape[0]
  K=numpy.exp(-D**2/eps_val)
  v = numpy.sqrt(numpy.sum(K,axis=0))
  
  A= K / numpy.outer(v,v)
  w=numpy.where(A > delta)
  Asp =  csc_matrix( (A[w],(w[0],w[1])), shape=A.shape )

  if neigen is None:
    neff = numpy.minimum(maxdim,n)
  else:
    neff =  numpy.minimum(neigen, n)

  from scipy.sparse.linalg import eigsh
  evals, evecs = eigsh(Asp, k=neff, which='LA', ncv=numpy.maximum(30,2*neff))
#  evecs= numpy.array([[1.,4,7],[2,5,8],[3,6,9],[5,10,15]]) 
  #evecs = matrix(c(1,2,3,5,4,5,6,10,7,8,9,15),nrow=4,ncol=3)
  psi = evecs[:,neff-1::-1]/numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)
  phi = evecs[:,neff-1::-1]*numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)

  eigenvals = evals[neff::-1]#eigenvalues

  if t<=0:
    lambd = eigenvals[1:]/(1-eigenvals[1:])
    lambd= numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.sum(lam < 0.05)
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen]
    X = psi[:,1:neigen+1]*lambd[:,0:neigen]
  else:
    lambd= eigenvals[1:]**t
    lambd=numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.sum(lam < 0.05)
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen]
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
