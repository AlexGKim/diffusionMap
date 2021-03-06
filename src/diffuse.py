#!/usr/bin/env python

'''Methods to compute diffusion maps in python.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']
__all__ = ['diffuse', 'nystrom']


#import gc
import numpy
import numpy as np
import scipy
from scipy.stats import norm
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()

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

    #dM=importr('diffusionMap')
    # R distance
    #xr=numpy.array(robjects.r.dist(d))
    # Python distance
    xr = scipy.spatial.distance.pdist(d, 'euclidean')
    del d
#    xr_arr=scipy.spatial.distance.squareform(xr)
    #dmap = dM.diffuse(xr_arr, **kwargs)
#    dmap = diffuse_py(xr_arr, **kwargs)
    dmap = diffuse_py(xr, **kwargs)
    #dmap=dict()
    #dmap['X']=dmap_.rx('X')
    #dmap['phi0']=dmap_.rx('phi0')
    #dmap['eigenvals']=dmap_.rx('eigenvals')
    #dmap['eigenmult']=dmap_.rx('eigenmult')
    #dmap['psi']=dmap_.rx('psi')
    #dmap['phi']=dmap_.rx('phi')
    #dmap['neigen']=dmap_.rx('neigen')
    #dmap['epsilon']=dmap_.rx('epsilon')
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
    
    #### NOTE Take advantage of the fact that orig and d are temp object


#     # Concatenate new data to training data

#     data_concat = np.concatenate((orig, d))
#     lenorig = len(orig)
# #    del(orig)
# #    del(d)

#     n_samp, n_feat = data_concat.shape
#     data_concat = data_concat.reshape(n_samp, 1, n_feat)
# #    data_concat_T = data_concat.reshape(1, n_samp, n_feat)

#     # Compute one big distance matrix
#     xr = scipy.spatial.distance.pdist(data_concat[:,0,:], 'euclidean')
#     distmat = scipy.spatial.distance.squareform(xr)    
# #    distmat = np.sum((data_concat_T - data_concat)**2, axis=2)
    
#     # Slice off the part of the distance matrix that 
#     # represents distances from new data to original data
#     # distmat_slice = distmat[len(orig):, :len(orig)]
#     distmat_slice = distmat[lenorig :, :lenorig ]


    distmat_slice = scipy.spatial.distance_matrix(d,orig)

    del d,orig

    # Compute nystrom and return

    # R version
    #coords = dM.nystrom(dmap, distmat_slice, **kwargs)
    # Python version
    coords = nystrom_py(dmap, distmat_slice, **kwargs)

    return coords

def nystrom_py(dmap, Dnew, **kwargs):

    Nnew, Nold  =  Dnew.shape
#    eigenvals=numpy.array(dmap.rx('eigenvals')[0])
    eigenvals=dmap.eigenvals
#    X=numpy.array(dmap.rx('X')[0])
    X=dmap.X
#    sigma=dmap.rx('epsilon')[0]
    sigma=dmap.epsilon
    if Nold != X.shape[0]:
      print "dimensions don't match"

    #decrease memory footprint taking advantage that Dnew is temp
    # Xnew = numpy.exp(-Dnew**2/sigma)
    numpy.exp(-Dnew**2/sigma, Dnew)
    Xnew = Dnew
    v = numpy.sum(Xnew,axis=1)

    #some points are far away from the training set
    w = numpy.where(v !=0)
    bw = numpy.where( v==0)

    Xnew[w,:]/=v[w,None]

    # for i in xrange(Xnew.shape[1]):
    #   #Xnew[:,i]=Xnew[:,i]/v
    #   Xnew[w,i]=Xnew[w,i]/v[w]


    dum = numpy.array(X)
    dum /= eigenvals[None,:]

    # for i in xrange(X.shape[1]):
    #   dum[:,i]=X[:,i]/eigenvals[i]

    Xnew = numpy.dot(Xnew ,dum)
    Xnew[bw,:]=numpy.nan
    return Xnew

def epsilonCompute(D, p=0.01):

   D=scipy.spatial.distance.squareform(D)

   n = D.shape[0]
   k = numpy.ceil(p*n)
   k = numpy.maximum(2,k)
   D_sort = numpy.sort(D,0)
   dist_knn = D_sort[k,:]
   epsilon = 2*numpy.median(dist_knn)**2
   return epsilon

class DiffuseOutput(object):
  __slots__ = ['X', 'phi0', 'eigenvals', 'eigenmult','psi','phi','neigen','epsilon']

  """docstring for DiffuseOutput"""
  def __init__(self, X, phi0, eigenvals, lambd,psi,phi,neigen,eps_val):
    super(DiffuseOutput, self).__init__()
    self.X=X
    self.phi0=phi0
    self.eigenvals=eigenvals
    self.eigenmult=lambd
    self.psi=psi
    self.phi=phi
    self.neigen=neigen
    self.epsilon=eps_val
    
def diffuse_py(D,eps_val='default',neigen=None,t=0,maxdim=50,delta=1e-5, var=0.68):
    #### NOTE Take advantage of the fact that D is a temp object

  if eps_val == 'default':
    eps_val = epsilonCompute(D)


  # original memory hog
  # K=numpy.exp(-D**2/eps_val)
  # v = numpy.sqrt(numpy.sum(K,axis=0))
  
  # A= K / numpy.outer(v,v)
  # del K

  # D_=scipy.spatial.distance.squareform(D)
  # n=D_.shape[0]
  # D_=numpy.exp(-D_**2/eps_val)
  # v = numpy.sqrt(numpy.sum(D_,axis=0))
  # D_=D_ / numpy.outer(v,v)

  # A=D_
  # w=numpy.where(A > delta)
  # Asp =  csc_matrix( (A[w],(w[0],w[1])), shape=A.shape )

  #D is a "distance matrix"
  n = int((1+numpy.sqrt(1+8.*D.shape[0])/2))


  # Rewrite code for memory not speed
  numpy.exp(-D**2/eps_val, D)
  v= numpy.zeros(n)
  for i in xrange(0,n):
    v[i]=1.
    a = numpy.arange(i,dtype='int')
    indeces = n*a - a*(a+1)/2 + i - 1 - a
    v[i] = v[i]+D[indeces].sum()
    a=numpy.arange(i+1,n,dtype='int')
    indeces=n*i - i*(i+1)/2 + a - 1 - i
    v[i] = v[i]+D[indeces].sum()
  numpy.sqrt(v,v)

  w=numpy.arange(D.shape[0],dtype='int')
  b = 1 -2*n 
  # is_ = (numpy.floor((-b - numpy.sqrt(b**2 - 8*w))/2)).astype('int',copy=False)
  # js_ = (w + is_*(b + is_ + 2)/2 + 1).astype('int',copy=False)
  # D=D/v[is_]/v[js_]
  is_ = (numpy.floor((-b - numpy.sqrt(b**2 - 8*w))/2)).astype('int',copy=False)
  D /= v[is_]
  is_ = (w + is_*(b + is_ + 2)/2 + 1).astype('int',copy=False)
  D /= v[is_]

  w=numpy.where(D > delta)[0]
  D=D[w]
  is_ = (numpy.floor((-b - numpy.sqrt(b**2 - 8*w))/2)).astype('int',copy=False)
  js_ = (w + is_*(b + is_ + 2)/2 + 1).astype('int',copy=False)

  D=numpy.append(D,D)
  tempis_=numpy.array(is_)
  is_=numpy.append(is_,js_)
  js_=numpy.append(js_,tempis_)
  del tempis_
  D=numpy.append(D,1/v/v)
  del v
  is_=numpy.append(is_,numpy.arange(n,dtype='int'))
  js_=numpy.append(js_,numpy.arange(n,dtype='int'))

  Asp =  csc_matrix( (D,(is_,js_)), shape=(n,n) )

  if neigen is None:
    neff = numpy.minimum(maxdim,n)
  else:
    neff =  numpy.minimum(neigen, n)
  neff=neff.item() #convert numpy int to int

  evals, evecs = eigsh(Asp, k=neff, which='LA', ncv=numpy.maximum(30,2*neff) )

  del D, Asp

  #don't bother with renormalization
  psi = evecs[:,neff-1::-1]
  phi = evecs[:,neff-1::-1]
  # psi = evecs[:,neff-1::-1]/numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)
  # phi = evecs[:,neff-1::-1]*numpy.outer(evecs[:,neff-1],numpy.zeros(neff)+1)

  eigenvals = evals[neff-1::-1]#eigenvalues

  if t<=0:
    lambd = eigenvals[1:]/(1-eigenvals[1:])
    lambd= numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.amax(numpy.where(lam > (1-var)))+1
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen+1]
    X = psi[:,1:neigen+1]*lambd[:,0:neigen]
  else:
    lambd= eigenvals[1:]**t
    lambd=numpy.outer(numpy.zeros(n)+1., lambd)
    if neigen is None:
      lam=lambd[0,:]/lambd[0,0]
      neigen= numpy.amax(numpy.where(lam > (1-var)))+1
      neigen = numpy.minimum(neigen,maxdim)
      eigenvals = eigenvals[0:neigen+1]
      X = psi[:,1:neigen+1]*lambd[:,0:neigen]
      y=DiffuseOutput(X,phi[:,0],eigenvals[1:],lambd[:,:neigen],psi,phi,neigen,eps_val)
  # y=dict()
  # y['X']=X
  # y['phi0']=phi[:,0]
  # y['eigenvals']=eigenvals[1:]
  # y['eigenmult']=lambd[:,:neigen]
  # y['psi']=psi
  # y['phi']=phi
  # y['neigen']=neigen
  # y['epsilon']=eps_val
  return y
