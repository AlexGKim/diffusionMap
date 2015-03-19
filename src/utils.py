#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'

import os
import os.path
import pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.colors
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy
from argparse import ArgumentParser
import scipy
from scipy.stats import norm
import sklearn
from sklearn.cross_validation import cross_val_score
import sklearn.ensemble
import sklearn.metrics.pairwise
import sklearn.grid_search
import sklearn.base
from sklearn.metrics import  make_scorer
import diffuse
import copy
from matplotlib.legend_handler import HandlerNpoints
from scipy.stats import norm
from scipy.stats.mstats import moment
#from guppy import hpy

matplotlib.rc('figure', figsize=(11,11))

"""
For the moment, the redshift range that we consider are set as global
variables.

The full redshift range of the sample is not used.  At high-redshift
the number of objects with spectroscopic redshifts is small making
the training and testing not robust.  The redshift range is based on
photometric redshifts since this is available to all galaxies.
"""
zmin = 0.1
zmax = 0.8

class Plots:
    @staticmethod
    def x(x,xlabel=None, figax='default',nsig=None,ndim=None, **kwargs):

        if ndim is None:
            ndim=x.shape[1]-1

        if figax == 'default':
            fig, axes = plt.subplots(nrows=ndim,ncols=ndim)#,sharex='col',sharey='row')
            #fig.tight_layout()
            fig.subplots_adjust(wspace = 0.15)
            fig.subplots_adjust(hspace = 0.15)
            for i in xrange(axes.shape[0]):
                for j in xrange(axes.shape[1]):
                    axes[i,j].set_visible(False)
        else:
            fig,axes = figax


        for ic in xrange(ndim):
            for ir in xrange(ic,ndim):
                axes[ir,ic].set_visible(True)
                axes[ir,ic].scatter(x[:,ic],x[:,ir+1],edgecolor='none',**kwargs)
                # print ic,ir+1
                # print x[0:2,ic],x[0:2,ir+1]                                            
                #axes[ir,ic].legend(prop={'size':6,'alpha':1})
                if xlabel is not None:
                    if ic==0:
                        axes[ir,ic].set_ylabel(xlabel[ir+1])
                    if ir==ndim-1:
                        axes[ir,ic].set_xlabel(xlabel[ic])
                if ic != 0:
                    axes[ir,ic].get_yaxis().set_visible(False)
                if ir !=ndim-1:
                    axes[ir,ic].get_xaxis().set_visible(False)

                if nsig is not None:
                    l = len(x[:,ic])
                    #l = w.sum()
                    xl = numpy.sort(x[:,ic])
                    xmn = xl[int(l*.5)]
                    xsd = xl[int(l*.05):int(l*.95)].std()
                    xl = numpy.sort(x[:,ir+1])
                    ymn = xl[int(l*.5)]
                    ysd = xl[int(l*.05):int(l*.95)].std()

                    axes[ir,ic].set_xlim(xmn-nsig*xsd,xmn+nsig*xsd)
                    axes[ir,ic].set_ylim(ymn-nsig*ysd,ymn+nsig*ysd)

        
        return fig, axes


class Data(object):

    """ Storage class for the galaxy data.  These are stored natively
    as arrays as the DM algorithms work with arrays.

    Parameters
    ----------
    x : '~numpy.ndarray'
       Coordinates.  Used as diffusion map coordinates
    y : '~numpy.ndarray'
       Redshift bias.  Used to define 'good' and 'bad' biases
    z : '~numpy.ndarray'
       Photometric redshift. Used to define redshift bins.  Photometric
       (rather than spectroscopic) redshifts are used because they
       are available for all galaxies.
    """
    __slots__ = ['x', 'y', 'z','xlabel','ylabel','zlabel','__weakref__']

    def __init__(self, x, y, z, xlabel = None, ylabel=None, zlabel=None):
        super(Data, self).__init__()
        self.x=x
        self.y=y
        self.z=z
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.zlabel=zlabel

    def __getitem__(self , index):
        return Data(self.x[index],self.y[index],self.z[index])

    def ndata(self):
        return len(self.y)

    def lessthan(self, thresh):
        return self.y < thresh

    def output(self,filename):
       f=open(filename,'w')
       for a,b in zip(self.x, self.y):
          f.write('{} {} {} {} {}\n'.format(a[0],a[1],a[2],a[3],b))
       f.close()

    def stats(self):
        print self.y.mean(), self.y.std(), len(self.y)

    def plot(self,logic=None,**kwargs):
 
        if logic is None:
            logic = lambda x : numpy.logical_not(numpy.zeros(len(self.x),dtype='b'))

        if 'xlabel' not in kwargs:
            kwargs['xlabel']= self.xlabel

        if 'alpha' not in kwargs:
            kwargs['alpha']= 0.025

        if 's' not in kwargs:
            kwargs['s']=5

        if 'marker' not in kwargs:
            kwargs['marker']= '.'

        figax = Plots.x(self.x[logic(self),:], **kwargs)


        return figax

    def plotClassification(self,thresh, data_predict, **kwargs):
        figax = self.plot(lambda x : numpy.logical_and(numpy.abs(x.y) <= thresh,data_predict),
            label='low bias',color='b',alpha=0.1,s=20, **kwargs)
        self.plot(lambda x : numpy.logical_and(numpy.abs(x.y) > thresh,data_predict),
           label='high bias',color='r',alpha=0.1,s=20,figax=figax,**kwargs)
        figax[0].suptitle('Classified True')

        figax2 = self.plot(lambda x : numpy.logical_and(numpy.abs(x.y) <= thresh,numpy.logical_not(data_predict)),
            label='low bias',color='b',alpha=0.1,s=20,**kwargs)
        self.plot(lambda x : numpy.logical_and(numpy.abs(x.y) > thresh,numpy.logical_not(data_predict)),
           label='high bias',color='r',alpha=0.1,s=20,figax=figax2,**kwargs)
        figax2[0].suptitle('Classified False')
        return figax,figax2


import resource
class DiffusionMap(object):

    """ Class that encapsulates a diffusion map.  A diffusion map is
    specified by coordinates x and parameters that dictate the
    behavior of the diffusion map algorithm.  Based on reading and
    discussions with Joey, the one free parameter to optimize is
    eps_val.

    Parameters
    ----------

    data : object of class Data
    par : double
       the value of eps_val

    """
    __slots__=['data','par', 'key', 'neigen','weight','dmap','neigen','__weakref__']

    def __init__(self, data, par, label=None):
        super(DiffusionMap, self).__init__()
        self.data = data
        self.par = par  #for the moment eps_val
        self.key=label
        self.weight= numpy.zeros(data.x.shape[1])+1
        self.dmap=None
        self.neigen=None

    def make_map(self):

        """ Method that calculates the diffusion map.  The result is
        stored internally.
        """

        kwargs=dict()
        kwargs['eps_val'] = self.par.item()
        kwargs['t']=1
        kwargs['delta']=1e-8
        kwargs['var']=0.95

        #### NOTE Take advantage of the fact that self.data.x*self.weight is a temp object
        self.dmap = diffuse.diffuse(self.data.x*self.weight,**kwargs)
        self.neigen = self.dmap['neigen']
#        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def transform(self, x):

        """ Method that calculates DM coordinates given input coordinates.

        Parameters
        ----------
        x: '~numpy.ndarray'
          Native coordinates for a set of points

        Returns
        -------
        '~numpy.ndarray'
          DM coordinates for a set of points
        """
        #### NOTE Take advantage of the fact that self.data.x*self.weight is a temp object

        return diffuse.nystrom(self.dmap, self.data.x*self.weight, x*self.weight)


    def plot(self,ax, oplot=False, **kwargs):
        X=self.dmap['X']
        ax.scatter(X.T[0],X.T[1],X.T[2],**kwargs)
        if not oplot:
            ax.set_xlabel('X[0]')
            ax.set_ylabel('X[1]')
            ax.set_zlabel('X[2]')

    def plot_external(self,ax, inx, oplot=False, **kwargs):
        X = self.transform(inx)
        ax.scatter(X.T[0],X.T[1],X.T[2],**kwargs)
        if not oplot:
            ax.set_xlabel('X[0]')
            ax.set_ylabel('X[1]')
            ax.set_zlabel('X[2]')

    def data_dm(self):
        return Data(self.dmap['X'],self.data.y,self.data.z,xlabel=[str(i) for i in xrange(self.neigen)])

    def data_mindist(self):
        dist = sklearn.metrics.pairwise_distances(self.data.x*self.weight,self.data.x*self.weight)
        numpy.fill_diagonal(dist,dist.max()) #numpy.finfo('d').max)
        return numpy.min(dist,axis=0)

class MyEstimator(sklearn.base.BaseEstimator):
    """docstring for MyEstimator"""

    __slots__=['catastrophe_cut','eps_par','mask_var','catastrophe', 'dm', 'max_distance','mask_scale','outlier_cut','optimize_frac',
    'xlabel','ylabel']

    #this is for dubugging memory
    # ncalls = 0

    def __init__(self, catastrophe_cut=numpy.float_(0.05), eps_par=numpy.float_(0.),
        mask_var=numpy.float_(1),xlabel=None,ylabel=None):
        super(MyEstimator, self).__init__()

        # self.params=dict()
        # self.params['catastrophe_cut']=catastrophe_cut
        # self.params['eps_par']=eps_par
        # self.params['mask_var']=mask_var

        #these are the parameters
        self.catastrophe_cut=catastrophe_cut
        self.eps_par=eps_par
        self.mask_var=mask_var

        self.catastrophe=None
        self.dm=None
        self.max_distance=None
        self.mask_scale=None

        self.outlier_cut=0.95
        self.optimize_frac = 0.1
        self.xlabel=xlabel
        self.ylabel=ylabel

        # MyEstimator.ncalls = MyEstimator.ncalls+1
        # if MyEstimator.ncalls ==4:
        #     print hp.heap()
        #     hp.setref()
        # elif MyEstimator.ncalls > 4:
        #     print hp.heap()

    def get_params(self,deep=True):
        params=dict()
        params['catastrophe_cut']=self.catastrophe_cut
        params['eps_par']=self.eps_par
        params['mask_var']=self.mask_var
        return params

    def set_params(self, catastrophe_cut=None, eps_par=None,
        mask_var=None):
        self.catastrophe_cut=catastrophe_cut
        self.eps_par=eps_par
        self.mask_var=mask_var

        return self

    def fit(self, x, y):
        del(self.catastrophe)
        del(self.max_distance)
        del(self.mask_scale)
        del(self.dm)

        self.catastrophe = numpy.abs(y) > self.catastrophe_cut



        if False: #os.path.isfile('estimator.pkl'):
            #print 'get pickle'
            pklfile=open('estimator.pkl','r')
            self.dm=pickle.load(pklfile)
        else:
            # the new coordinate system based on the training data
            data = Data(x,y,numpy.zeros(len(y)),xlabel=self.xlabel,ylabel=self.ylabel)
            self.dm=DiffusionMap(data,self.eps_par)
            mindist = self.dm.data_mindist()
            mindist= numpy.log(mindist)
            mu, std = norm.fit(mindist)
            wok=numpy.abs(mindist-mu)/std < 3
            mu, std = norm.fit(mindist[wok])
            self.dm.par = numpy.exp(mu+self.eps_par*std)
            self.dm.make_map()

        #     pklfile=open('estimator.pkl','w')
        #     pickle.dump(self.dm,pklfile)
        # pklfile.close()
        # self.dm=DiffusionMap(x,self.eps_par)
        # self.dm.make_map()


        train_dist = sklearn.metrics.pairwise_distances(self.dm.data_dm().x,self.dm.data_dm().x)
        # catastrophe_distances = train_dist[numpy.outer(self.catastrophe,self.catastrophe)]
        # catastrophe_distances = catastrophe_distances[catastrophe_distances !=0]
        # catastrophe_distances = numpy.sort(catastrophe_distances)
        numpy.fill_diagonal(train_dist,train_dist.max()) #numpy.finfo('d').max)
        train_min_dist = numpy.min(train_dist,axis=0)
        train_min_dist = numpy.sort(train_min_dist)
        catastrophe_min_dist = train_min_dist[self.catastrophe] 
        catastrophe_min_dist=numpy.log(catastrophe_min_dist)
        mu, std = norm.fit(catastrophe_min_dist)
        wok=numpy.abs(catastrophe_min_dist-mu)/std < 3
        mu, std = norm.fit(catastrophe_min_dist[wok])
        self.max_distance = train_min_dist[x.shape[0]*self.outlier_cut]
        self.mask_scale = numpy.exp(mu+std*self.mask_var)


    def predict(self, x):
        return 0

    def weight(self,x):
        x_dm = self.dm.transform(x)

        test_dist = sklearn.metrics.pairwise_distances(self.dm.data_dm().x,x_dm)   
        test_min_dist = numpy.min(test_dist,axis=0)


        closer  = test_min_dist < self.max_distance
        return numpy.exp(-(test_dist[self.catastrophe,:]/self.mask_scale)**2).sum(axis=0),closer
    
    def score(self,x,y):
        # x_dm = self.dm.transform(x)

        # test_dist = sklearn.metrics.pairwise_distances(self.dm.data_dm().x,x_dm)   
        # test_min_dist = numpy.min(test_dist,axis=0)

        # closer  = test_min_dist < self.max_distance

        weight, closer = self.weight(x)
        weight_sort=numpy.sort(weight[closer])
        w=weight[closer] <= weight_sort[self.optimize_frac*len(weight_sort)]
#        w=numpy.logical_and(w,closer)
        ans=-moment(y[closer[w]],moment=4).item()
        sd= y[closer[w]].std()
        print self.catastrophe_cut,self.eps_par,self.mask_var,ans,sd
        return ans

    def plots(self,x):

        figax= train_data.plot(color='b',alpha=0.1,s=10)
        train_data.plot(lambda x: self.catastrophe, color='r',alpha=1,s=20,figax=figax)
        plt.savefig('outliers.png')

        for i in xrange(6):
            crap = numpy.sort(self.dm.data_dm().x[:,i])
            crap= crap[len(crap)*.1:len(crap)*.9]
            sig = crap.std()
            cm=matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=crap[len(crap)/2]-5*sig,
                    vmax=crap[len(crap)/2]+5*sig),cmap='Spectral')
            cval=cm.to_rgba(self.dm.data_dm().x[:,i])
            figax= train_data.plot(c=cval,alpha=0.3,s=20,cmap=cm)
            figax[0].suptitle(str(i))
            plt.savefig('splits.'+str(i)+'.png')

        figax= self.dm.data_dm().plot(color='r',alpha=0.1,s=10,ndim=6)
        self.dm.data_dm().plot(lambda x: self.catastrophe,
            color='b',alpha=0.1,s=20,ndim=6,figax=figax)
        plt.savefig('temp.png')
        figax= self.dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
        self.dm.data_dm().plot(lambda x: self.catastrophe,
            color='b',alpha=0.2,s=20,ndim=6,figax=figax)
        plt.savefig('temp2.png')

        cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
        cval=cm.to_rgba(self.weight(x))
        figax= x.plot(c=cval,alpha=0.2,s=20,cmap=cm,vmin=0,vmax=cval.max())
        plt.savefig('color_dm.png')

