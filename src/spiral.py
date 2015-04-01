#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

import pyfits
import pickle
import diffuse
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def scatter(X, bias, marker='+', label=None, size=20):
    '''Add a scatter plot to the current figure. 
    
    Parameters
    ----------
    
    X:  diffusion  coordinate  representation  of dataset.  First  two
       coordinates are used.

    bias: photoz bias of X.
    
    marker: the marker to use on the figure.
    '''
    plt.scatter(X.T[0], X.T[1], c=bias,
                vmin=-.1, vmax=.1,
                cmap=plt.matplotlib.cm.jet, 
                marker=marker, label=label,s=size)

def plot_dmap(dmap, fname, bias, nystrom=None, nystrombias=None):
    '''After computing the diffusion map for training data, plot the results.
    Also plot nystrom if computed.

    Parameters
    ----------
    dmap: The trained diffusion map to plot.
    
    fname: The name of the PDF file in which to save the plot. 

    nystrom: ndarray of shape (n_samples, 2), or None (default). If
      not None, the first two diffusion coordinates of out-of-sample
      testing data. 
    '''
    plt.clf()
    coords = np.array(dmap.rx('X')[0])
    coords = dmap['X']
    plt.xlabel('diffusion coordinate 1')
    plt.ylabel('diffusion coordinate 2')
    plt.title('RedMaGiC Diffusion Map: Absolute Magnitudes Only')
    scatter(coords, bias, marker='.')
    plt.colorbar(label='bias')
    if nystrom and nystrombias:
        scatter(nystrom, nystrombias)
    if fname is not None:
        plt.savefig(fname ,format='pdf')
    

if __name__ == '__main__':

    # training data
    Norig = 4000
    Next = 1000
    np.random.seed(0)
    t=np.random.uniform(size=Norig+Next)**.7*10
#    t=runif(Norig+Next)^.7*10
    al=.15
    bet=.5
    data = np.zeros((Norig+Next,2))
    data[:,0]=bet*np.exp(al*t)*np.cos(t)+np.random.normal(size=Norig+Next,loc=0,scale=.1)
    data[:,1]=bet*np.exp(al*t)*np.sin(t)+np.random.normal(size=Norig+Next,loc=0,scale=.1)

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('spiral2.pdf')
    plt.clf()
    plt.scatter(data[0:Norig,0],data[0:Norig,1],marker='D',c=t[0:Norig], cmap=plt.cm.hsv)
    plt.scatter(data[Norig:,0],data[Norig:,1],marker='^',c=t[Norig:], cmap=plt.cm.hsv)
    plt.savefig(pp,format='pdf')
    from mpl_toolkits.mplot3d import Axes3D

    ts =[1]
    for tt in ts:
      plt.clf()
    # Train diffusion maps
      dmap = diffuse.diffuse(data[0:Norig,:], t=tt)

    # Nystrom
      train = np.array(diffuse.nystrom(dmap, data[0:Norig,:], data[Norig:,:]))
      fig = plt.figure()
      ax = fig.add_subplot(111)
      #X=np.array(dmap.rx('X')[0])
      X=dmap['X']
      ax.scatter(X.T[0],X.T[1],marker='D',c=t[0:Norig], cmap=plt.cm.hsv)
      X=train
      ax.scatter(X.T[0],X.T[1],marker='^',c=t[Norig:], cmap=plt.cm.hsv)
      ax.set_xlabel('X[0]')
      ax.set_ylabel('X[1]')
      ax.set_title('t='+str(tt))
      plt.savefig(pp, format='pdf')
    
    pp.close()
