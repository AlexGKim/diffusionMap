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
#from mpi4py import MPI
from utils import *
import sys

#from guppy import hpy

matplotlib.rc('figure', figsize=(11,11))


""" Method that reads in input data, selects those appropriate for
    training and testing, and creates training and test subsets.

    Parameters
    ----------
    test_size: float
      Fraction of appropriate data reserved for test set
    random_state: int or RandomState
      Specifies the random assignment to train and test sets
"""

def manage_data(test_size=0.1, random_state=7):
    # Load training data
    dir='/project/projectdirs/desi/target/analysis/deep2/v1.0/'
    filenames = ['deep2egs-oii.fits.gz',  'deep2egs-phot.fits.gz',  'deep2egs-stars.fits.gz']

    keys=['CFHTLS_G','CFHTLS_R','CFHTLS_Z','ZHELIO','CFHTLS_I','RA','DEC']

    import pyfits
    f = pyfits.open(dir+filenames[1])
    photdata = f[1].data
    f2 = pyfits.open(dir+filenames[2])
    stardata = f2[1].data

    notn99_p = photdata[keys[2]]!=-99.
    notn99_p = numpy.logical_and(notn99_p, photdata[keys[1]]!=-99.)
    notn99_p = numpy.logical_and(notn99_p, photdata[keys[0]]!=-99.)

    notn99_s = stardata[keys[2]]!=-99.
    notn99_s = numpy.logical_and(notn99_s, stardata[keys[1]]!=-99.)
    notn99_s = numpy.logical_and(notn99_s, stardata[keys[0]]!=-99.)

    gmz_p= (photdata[keys[0]] - photdata[keys[2]])[notn99_p]
    gmr_p= (photdata[keys[0]] - photdata[keys[1]])[notn99_p]
    rmz_p= (photdata[keys[1]] - photdata[keys[2]])[notn99_p]

    temp = rmz_p.reshape(-1,1)
    temp = numpy.hstack((temp,gmz_p.reshape(-1,1)))
    temp = numpy.hstack((temp,rmz_p.reshape(-1,1)))
    objno_p = photdata['OBJNO'][notn99_p]

    gmz_s= (stardata[keys[0]] - stardata[keys[2]])[notn99_s]
    gmr_s= (stardata[keys[0]] - stardata[keys[1]])[notn99_s]
    rmz_s= (stardata[keys[1]] - stardata[keys[2]])[notn99_s]

    features = numpy.append(rmz_p, rmz_s).reshape(-1, 1)
    features = numpy.hstack((features, numpy.append(gmz_p, gmz_s).reshape(-1, 1)))
    features = numpy.hstack((features, numpy.append(gmr_p, gmr_s).reshape(-1, 1)))

    f3 = pyfits.open(dir+filenames[0])
    oiidata = f3[1].data

    photredshifts = numpy.zeros(len(objno_p))
    photoii = numpy.zeros(len(objno_p))
    for objno, redshift,oiiflux in zip(oiidata['OBJNO'],oiidata['ZBEST'],oiidata['OII_3727']):
        w = objno_p == objno
        photredshifts[w] = redshift
        photoii[w] = oiiflux

    redshifts = numpy.append(photredshifts, numpy.zeros(notn99_s.sum()))
    oiifluxes = numpy.append(photoii, numpy.zeros(notn99_s.sum()))

    from sklearn.cross_validation import train_test_split
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, z_train, z_test = \
                                        train_test_split(features,
                                                         redshifts,
                                                         oiifluxes,
                                                         test_size=test_size,
                                                         random_state=random_state)

    return Data(X_train, y_train, z_train,xlabel=['g-z','r-z','g-r'], ylabel='redshifts',zlabel='oiifluxes'), Data(X_test,  y_test, z_test,xlabel=['g-z','r-z','g-r'], ylabel='redshifts',zlabel='oiifluxes')

class MyEstimator(sklearn.base.BaseEstimator):
    """docstring for MyEstimator"""

    __slots__=['eps_par','mask_var','catastrophe', 'dm', 'max_distance','mask_scale','outlier_cut','optimize_frac',
    'xlabel','ylabel']

    #this is for dubugging memory
    # ncalls = 0

    def __init__(self, eps_par=numpy.float_(0.),
        mask_var=numpy.float_(1),xlabel=None,ylabel=None):
        super(MyEstimator, self).__init__()

        # self.params=dict()
        # self.params['']=
        # self.params['eps_par']=eps_par
        # self.params['mask_var']=mask_var

        #these are the parameters
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

        self.zmin = 0.6
        self.zmax=1.6

        self.oiimin=6e-17

        # MyEstimator.ncalls = MyEstimator.ncalls+1
        # if MyEstimator.ncalls ==4:
        #     print hp.heap()
        #     hp.setref()
        # elif MyEstimator.ncalls > 4:
        #     print hp.heap()

    def get_params(self,deep=True):
        params=dict()
        params['eps_par']=self.eps_par
        params['mask_var']=self.mask_var
        return params

    def set_params(self, eps_par=None, mask_var=None):
        self.eps_par=eps_par
        self.mask_var=mask_var

        return self

    def fit(self, x, y):
        del(self.catastrophe)
        del(self.max_distance)
        del(self.mask_scale)
        del(self.dm)

        self.catastrophe = numpy.logical_and(y[:,0] > self.zmin,
            numpy.logical_and(y[:,0]<self.zmax,y[:,1] > self.oiimin))



        if False: #os.path.isfile('estimator.pkl'):
            #print 'get pickle'
            pklfile=open('estimator.pkl','r')
            self.dm=pickle.load(pklfile)
        else:
            # the new coordinate system based on the training data
            data = Data(x,y,numpy.zeros(len(y)),xlabel=self.xlabel,ylabel=self.ylabel)
            self.dm=DiffusionMap(data,self.eps_par)
            mindist = self.dm.data_mindist()
            mindist[mindist < sys.float_info.min]=(mindist[mindist > sys.float_info.min]).min()
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
        train_min_dist[train_min_dist < sys.float_info.min]=(train_min_dist[train_min_dist > sys.float_info.min]).min()
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
        weight_sort = weight_sort[::-1]
        w=weight[closer] >= weight_sort[self.optimize_frac*len(weight_sort)]
#        w=numpy.logical_and(w,closer)
        ans=-moment(y[closer[w]],moment=4).item()
        sd= y[closer[w]].std()
        print self.eps_par,self.mask_var,ans,sd
        return ans

    def plots(self,x):

        figax= train_data.plot(color='r',alpha=0.1,s=10)
        train_data.plot(lambda x: self.catastrophe, color='b',alpha=0.1,s=10,figax=figax)
        plt.savefig('../results/outliers.png')

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
            plt.savefig('../results/splits.'+str(i)+'.png')

        figax= self.dm.data_dm().plot(color='r',alpha=0.1,s=10,ndim=6)
        self.dm.data_dm().plot(lambda x: self.catastrophe,
            color='b',alpha=0.1,s=20,ndim=6,figax=figax)
        plt.savefig('../results/temp.png')
        figax= self.dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
        self.dm.data_dm().plot(lambda x: self.catastrophe,
            color='b',alpha=0.2,s=20,ndim=6,figax=figax)
        plt.savefig('../results/temp2.png')

        cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
        cval=cm.to_rgba(self.weight(x.x))
        figax= x.plot(c=cval,alpha=0.2,s=20,cmap=cm,vmin=0,vmax=cval.max())
        plt.savefig('../results/color_dm.png')


if __name__ == '__main__':

    doplot = False

    parser = ArgumentParser()
    parser.add_argument('cv', nargs='?',default=3,type=int)
    parser.add_argument('n_jobs', nargs='?',default=1,type=int)
    parser.add_argument('test_size', nargs='?',default=0.2,type=float)
    parser.add_argument('seed', nargs='?',default=9)
    parser.add_argument('--test',default=False,type=bool)
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    x0 = numpy.array([-2.,0.])

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)

    # figax= train_data.plot(color='b',alpha=0.1,s=10)
    # train_data.plot(lambda x: numpy.logical_and(train_data.y<1.6,numpy.logical_and(train_data.y> 0.6,train_data.z>8e-17)) , color='r',alpha=0.1,s=10,figax=figax)
    # #plt.savefig('outliers.png')
    # plt.show()

    estimator = MyEstimator(eps_par=x0[0],mask_var=x0[1],xlabel=train_data.xlabel,ylabel=train_data.ylabel)
    fity = numpy.hstack((train_data.y.reshape(-1,1),train_data.z.reshape(-1,1)))
    print 'fitting'
    estimator.fit(train_data.x, fity)
    print 'done fitting'
    # estimator.score(test_data, test_data.y)
    estimator.plots(test_data)

    wef

    # optimize
    if pdict['test']:
      param_grid = [{'': numpy.arange(0.03,.1,0.05), 'eps_par': numpy.arange(-2,2,10),
       'mask_var': numpy.arange(-2,2.1,10)}]
    else:
      param_grid = [{'': numpy.arange(0.03,.1,0.01), 'eps_par': numpy.arange(-3,3.01,1),
      'mask_var': numpy.arange(-3,3.01,1)}]

    print pdict

    from sklearn.externals import joblib
    #filename = '../results/clf_mpi.pkl'
    if False: #os.path.isfile(filename):
        clf = joblib.load(filename) 
        #print 'get pickle'
        # pklfile=open(filename,'r')
        # clf=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        del(test_data)

        me = MPI.COMM_WORLD.Get_rank()

        param_grid[0]['']=numpy.array([param_grid[0][''][me]])
        print me, param_grid

        clf = sklearn.grid_search.GridSearchCV(estimator, param_grid, n_jobs=pdict['n_jobs'],
            cv=pdict['cv'],pre_dispatch='n_jobs',refit=True)
        clf.fit(train_data.x,train_data.y)

        clf = MPI.COMM_WORLD.gather(clf,root=0)
 
        if me==0:
            best_score_= -1e10
            for cl in clf:
                print cl.best_score_, cl.best_params_
                if cl.best_score_ > best_score_:
                    best_score_= cl.best_score_
                    bestcl=cl
            joblib.dump(bestcl, filename) 
#        pklfile=open(filename,'w')
#        pickle.dump(clf,pklfile)
#        pklfile.close()
        import sys
        sys.exit()


    print clf.best_params_
    print clf.score(test_data.x,test_data.y)

