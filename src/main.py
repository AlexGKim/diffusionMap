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
#from matplotlib.backends.backend_pdf import PdfPages
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
    import pyfits
    f = pyfits.open('../data/stripe82_run_redmagic-1.0-08.fits')
    data = f[1].data

    # Get rid of entries without spectroscopic redshifts.
    inds = data['ZSPEC'] > 1e-2
    sdata = data[inds]

    # Get rid of entries outside of photo-z range
    inds = numpy.logical_and(sdata['ZRED2'] >= zmin, sdata['ZRED2'] < zmax)
    sdata = sdata[inds]

    # Compute bias
    bias = sdata['ZRED2'] - sdata['ZSPEC']

    # Get features for *entire* sample.
    # Use G-R, R-I, I-Z colors and I absolute magnitude as features.
    features = sdata['MABS'][:, :-1] - sdata['MABS'][:, 1:] # colors
    features = numpy.hstack((features, sdata['MABS'][:, 2].reshape(-1, 1))) # i magnitude

    # Some cuts based on outliers
    inds = features[:,0] <4
    inds = numpy.logical_and(inds, features[:,1]<4)
    inds = numpy.logical_and(inds, features[:,2] > 0.1)

    sdata=sdata[inds]
    bias=bias[inds]
    features=features[inds]

    # Scale features
#    from sklearn.preprocessing import StandardScaler
#    scaler = StandardScaler()
#    features_scaled = scaler.fit_transform(features)

    from sklearn.cross_validation import train_test_split
    # Split into training and testing sets
    X_train, X_test, y_train, y_test, z_train, z_test = \
                                        train_test_split(features,
                                                         bias,
                                                         sdata['ZRED2'],
                                                         test_size=test_size,
                                                         random_state=random_state)

    return Data(X_train, y_train, z_train,xlabel=['g-r','r-i','i-z','i'], ylabel='bias',zlabel='photo-z'), Data(X_test,  y_test, z_test,xlabel=['g-r','r-i','i-z','i'], ylabel='bias',zlabel='photo-z')


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
        self.weight= numpy.array([2,2,1,1])
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

#hp = hpy()

if __name__ == '__main__':

    doplot = False

    parser = ArgumentParser()
    parser.add_argument('me', nargs='?',default=0,type=int)
    parser.add_argument('cv', nargs='?',default=3,type=int)
    parser.add_argument('n_jobs', nargs='?',default=1,type=int)
    parser.add_argument('test_size', nargs='?',default=0.1,type=float)
    parser.add_argument('seed', nargs='?',default=9)
    parser.add_argument('--test',default=False,type=bool)
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    x0 = numpy.array([0.05,0.,0.05])

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)
    estimator = MyEstimator(catastrophe_cut=x0[0],
        eps_par=x0[1],mask_var=x0[2],xlabel=train_data.xlabel,ylabel=train_data.ylabel)



#    estimator.fit(train_data.x, train_data.y)
    # estimator.score(test_data, test_data.y)
    # estimator.plots(test_data.x)
    # optimize
    if pdict['test']:
      param_grid = [{'catastrophe_cut': numpy.arange(0.03,.1,0.05), 'eps_par': numpy.arange(-2,2,10),
       'mask_var': numpy.arange(-2,2.1,10)}]
    else:
      param_grid = [{'catastrophe_cut': numpy.arange(0.03,.1,0.01), 'eps_par': numpy.arange(-3,3.01,1),
      'mask_var': numpy.arange(-3.,3,1)}]

    from sklearn.externals import joblib
    filename = os.environ['SCRATCH']+'/diffusionMap/results/clf_mpi.'+str(pdict['me'])+'.pkl'
    if False: #os.path.isfile(filename):
        clf = joblib.load(filename) 
        #print 'get pickle'
        # pklfile=open(filename,'r')
        # clf=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        del(test_data)

        nmask=2


        me1=pdict['me']/len(param_grid[0]['eps_par'])/(len(param_grid[0]['mask_var'])/nmask)
        me2 = (pdict['me'] /(len(param_grid[0]['mask_var'])/nmask)) %len(param_grid[0]['eps_par'])
        me3 = (pdict['me'] %  (len(param_grid[0]['mask_var'])/nmask))*nmask

   
        param_grid[0]['catastrophe_cut']=numpy.array([param_grid[0]['catastrophe_cut'][me1]])
        param_grid[0]['eps_par']=numpy.array([param_grid[0]['eps_par'][me2]])
        param_grid[0]['mask_var']=param_grid[0]['mask_var'][me3:me3+nmask]

        clf = sklearn.grid_search.GridSearchCV(estimator, param_grid, n_jobs=pdict['n_jobs'],
            cv=pdict['cv'],pre_dispatch='n_jobs',refit=True)
        clf.fit(train_data.x,train_data.y)

        print 'result', pdict['me'], param_grid , clf.best_params_, clf.best_score_

        joblib.dump(clf, filename) 

        import sys
        sys.exit()


    print clf.best_params_
    print clf.score(test_data.x,test_data.y)
    #get distances
 






#     class OneOutput(object):
#         """docstring for OneOutput"""
#         def __init__(self, feature_importances_,y,dy2):
#             super(OneOutput, self).__init__()
#             self.feature_importances_ = feature_importances_
#             self.y=y
#             self.dy2=dy2
            
#     class Output(object):
#         """docstring for Output"""
#         def __init__(self,test_min_dist,train_cut, outputs):
#             super(Output, self).__init__()
#             self.test_min_dist=test_min_dist
#             self.train_cut=train_cut
#             self.outputs=outputs
#     filename='output.pkl'

#     if os.path.isfile(filename):
#         #print 'get pickle'
#         pklfile=open(filename,'r')
#         coloroutput,dmoutput=pickle.load(pklfile)
#     else:
#         nrealize=100
#         train_dist = sklearn.metrics.pairwise_distances(train_data.x,train_data.x)
#         numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
#         train_min_dist=numpy.min(train_dist,axis=0)
#         train_sort = numpy.argsort(train_min_dist)
#         train_sort = train_sort[0:prunefrac * len(train_sort)]
#         train_cut =  train_min_dist[train_sort[-1]]
#         test_dist = sklearn.metrics.pairwise_distances(train_data.x[train_sort],test_data.x)
#         test_min_dist = numpy.min(test_dist,axis=0)

#         clf = sklearn.ensemble.forest.RandomForestRegressor(n_estimators=100,random_state=12)
#         color_outs=[]
#         for rs in xrange(nrealize):
#             clf.set_params(random_state=rs)
#             clf.fit(train_data.x[train_sort],train_data.y[train_sort])
#             y,dy2 = clf.predict(test_data.x)
#             color_outs.append(OneOutput(clf.feature_importances_,y,dy2))
#         coloroutput=Output(test_min_dist,train_cut,color_outs)

#         train_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,0:4],dmsys.dmdata.x[:,0:4])**2+
#             sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,4:],dmsys.dmdata.x[:,4:])**2)
#         numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
#         train_min_dist=numpy.min(train_dist,axis=0)
#         train_sort = numpy.argsort(train_min_dist)
#         train_sort = train_sort[0:prunefrac * len(train_sort)]
#         train_cut =  train_min_dist[train_sort[-1]]
#         test_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,0:4],
#             test_data_dm.x[:,0:4])**2+sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,4:]
#             ,test_data_dm.x[:,4:])**2)
#         test_min_dist = numpy.min(test_dist,axis=0)

#         dm_outs=[]
#         for rs in xrange(nrealize):
#             clf.set_params(random_state=rs)
#             clf.fit(dmsys.dmdata.x[train_sort],dmsys.dmdata.y[train_sort])
#             y,dy2 = clf.predict(test_data_dm.x)
#             dm_outs.append(OneOutput(clf.feature_importances_,y,dy2))
#         dmoutput=Output(test_min_dist,train_cut,dm_outs)

#         pklfile=open(filename,'w')
#         pickle.dump([coloroutput,dmoutput],pklfile)
#     pklfile.close()


#     # w  = dmoutput.test_min_dist < dmoutput.train_cut
#     # plt.scatter(dmoutput.test_min_dist, dmoutput.outputs[0].y-test_data_dm.y)
#     # plt.xlim(0,dmoutput.train_cut*2)
#     # plt.show()

#     w  = coloroutput.test_min_dist < coloroutput.train_cut
#     plt.scatter(coloroutput.test_min_dist, coloroutput.outputs[0].y-test_data_dm.y)
#     plt.xlim(0,coloroutput.train_cut*2)
#     plt.show()

#     plt.scatter(dmoutput.outputs[0].dy2[w],dmoutput.outputs[0].y[w]-test_data_dm.y[w])
#     plt.show()

#     wef

#     if doplot:      
#         plt.clf()
#         figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         plt.clf()
#         figax= test_data_dm.plot(label='0.3',color='k',alpha=0.1,s=10,nsig=4)
#         test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2),nsig=4, label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1),nsig=4 , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         plt.clf()
#         figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         figax= test_data_dm.plot(label='0.3',color='k',nsig=4,alpha=0.1,s=10)
#         test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',nsig=4,alpha=1,s=10,figax=figax)
#         test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',nsig=4,alpha=1,s=10,
#             figax=figax)
#         plt.show()

#     c1,c2,c3,c4,c5, c6, c7 = meanuncertainties(test_data,coloroutput)
#     d1,d2,d3,d4,d5,d6,d7= meanuncertainties(test_data_dm,dmoutput)
#     # ind=0
#     # plt.clf()
#     # fig=plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.plot(c6[ind,:],label='c')
#     # ax.plot(d6[ind,:],label='d')
#     # plt.legend()
#     # plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.errorbar(c1,c2,yerr=c3,label='color',marker='o',color='b')
#     ax.errorbar(d1,d2,yerr=d3,label='dm',marker='o',color='r')
#     plt.legend()
#     plt.show()
#     fwe
#     # for a,b,c, d, e in zip(frac_include,mns.mean(axis=1),mns.std(axis=1),means,stds):
#     #     print "{:5.3f} {:7.4f} {:6.4f} {:6.4f} {:6.4f}".format(a,b,c, d, e)



#     # fwe
#     if doplot:
#         for index in xrange(8):
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.scatter(dmsys.dmdata.x[:,index],dmsys.dmdata.y)
#             ax.set_xlim((-20,20))
#         plt.show()
#     import matplotlib.collections
#     import matplotlib.lines
#     import matplotlib.patches
#     if doplot:
#         ## stuff to force the legend to have alpha=1
#         labs=['positive bias','negative bias','no bias']
#         # lines = [matplotlib.lines.Line2D([],[],color='r',marker='.',linestyle='None'),
#         # matplotlib.lines.Line2D([],[],color='b',marker='.',linestyle='None')
#         # ,matplotlib.lines.Line2D([],[],color='k',marker='.',linestyle='None')]
#         lines = [matplotlib.patches.Circle([], color='r'),matplotlib.patches.Circle([], color='b')
#             ,matplotlib.patches.Circle([], color='k')]

#         plt.clf()
#        # w = numpy.logical_and(split(train_data.x), numpy.abs(train_data.y) > x0[0])                    
#         figax= train_data.plot(lambda x: train_data.y > x0[0] , label='positive bias',color='r',alpha=0.02)
#         train_data.plot(lambda x: train_data.y < -x0[0] , label='negative bias',color='b',figax=figax,alpha=0.02)
#         train_data.plot(lambda x: numpy.abs(train_data.y) <= x0[0] , label='no bias',color='k',
#             figax=figax,alpha=0.02)

#         for ax in figax[1]:
#             for a in ax:
#                 a.legend(lines, labs,prop={'size':6})

#         plt.savefig('../results/colorspace.png')

#         plt.clf()
#         figax = dmsys.dmdata.plot(lambda x: dmsys.dmdata.y > x0[0] ,
#             label='positive bias',color='r',nsig=4,alpha=0.01)
#         dmsys.dmdata.plot(lambda x: dmsys.dmdata.y < -x0[0] ,
#             label='negative bias',color='b',figax=figax,nsig=4,alpha=0.01)
#         dmsys.dmdata.plot(lambda x: numpy.abs(dmsys.dmdata.y) <= x0[0] ,
#             label='no bias',color='k',figax=figax,nsig=4,alpha=0.01)

#         for ax in figax[1]:
#             for a in ax:
#                 a.legend(lines, labs,prop={'size':3})
#         plt.savefig('../results/dmspace.png')
     




#     # clf = objective_dm.clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)
#     # test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
#     # y_pred, y_var = clf.predict(test_data_dm.x)
#     # y_sig=numpy.sqrt(y_var)
#     # delta_y = y_pred-test_data.y
#     # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)
#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)
#     # ax1.legend()
#     # ax2.legend()
#     # plt.show()
    

#   # end old code
#     test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
#     dists=numpy.sqrt(numpy.dot(dmsys.dmdata.x[:,0:4],numpy.transpose(test_data_dm.x[:,0:4]))**2 +
#         numpy.dot(dmsys.dmdata.x[:,4:],numpy.transpose(test_data_dm.x[:,4:]))**2)
#     dists = numpy.max(dists,axis=0)
#     w=dists < 1e15

#     # if True:
#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(111)
#     #     objective_dm.plot_scatter_external(ax,test_data_dm.x[w],test_data_dm.y[w],label='dm',color='b')
#     #     objective_color.plot_scatter_external(ax,test_data.x[w],test_data.y[w],label='color',color='r')
#     #     objective_dm.plot_scatter_external(ax,dmsys.dmdata.x[w],dmsys.dmdata.y[w],label='dm',color='b',marker='x')
#     #     objective_color.plot_scatter_external(ax,train_data.x[w],train_data.y[w],label='color',color='r',marker='x')

#     #     ax.legend()
#     #     plt.show()
#     #     wefwe

#     if True:
#         plt.clf()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
# #        ax.scatter(objective_dm.frac_include,objective_dm.fvals,label='dm',color='b')
# #        ax.scatter(objective_color.frac_include,objective_color.fvals,label='color',color='r')
#         objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b',marker='x')
#         objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r',marker='x')
#         ax.legend()
#         plt.savefig('../results/sigmas.png')
#         wefwe


#     if doplot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         objective_dm.plot_scatter(ax,label='Train dm',color='b')
#         objective_color.plot_scatter(ax,label='Train color',color='r')
#         ax.legend()

#     if doplot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b')
#         objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r')


#         ax.legend()
#         plt.show()

#     # clf = clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)

#     # y_pred, y_var = clf.predict(dmsys.dmdata.x)
#     # y_sig=numpy.sqrt(y_var)

#     # delta_y = y_pred-dmsys.dmdata.y

#     # fig1 = plt.figure()

#     # ax1 = fig1.add_subplot(111)
#     # ax1.scatter(y_sig, delta_y, label='train')

#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # fig2 = plt.figure()
#     # ax2 = fig2.add_subplot(111)
#     # ax2.scatter(y_sig[predind[1:]],std,label='train')


#     # 
#     # y_pred, y_var = clf.predict(test_data_dm.x)
#     # y_sig=numpy.sqrt(y_var)

#     # delta_y = y_pred-test_data.y

#     # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)

#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)

#     # ax1.legend()
#     # ax2.legend()

#     # plt.show()

# def old():




#     dm=DiffusionMap(train_data,x0[1])



#     if os.path.isfile('dmsys.pkl'):
#         #print 'get pickle'
#         pklfile=open('dmsys.pkl','r')
#         dm,test_data_dm=pickle.load(pklfile)
#     else:
#         # the new coordinate system based on the training data
#         dm=DiffusionMap(train_data,x0[1])
#         dm.make_map()
#         test_data_dm=Data(dm.transform(test_data.x),test_data.y,test_data.z,
#             xlabel=[str(i) for i in xrange(dm.neigen)])
#         pklfile=open('dmsys.pkl','w')
#         pickle.dump([dm,test_data_dm],pklfile)
#     pklfile.close()


#     ## plots that show dm x in color space
#     if doplot:
#         for i in xrange(6):
#             crap = numpy.sort(dm.data_dm().x[:,i])
#             crap= crap[len(crap)*.1:len(crap)*.9]
#             sig = crap.std()
#             cm=matplotlib.cm.ScalarMappable(
#                 norm=matplotlib.colors.Normalize(vmin=crap[len(crap)/2]-5*sig,
#                     vmax=crap[len(crap)/2]+5*sig),cmap='Spectral')
#             cval=cm.to_rgba(dm.data_dm().x[:,i])
#             figax= train_data.plot(c=cval,alpha=0.3,s=20,cmap=cm)
#             figax[0].suptitle(str(i))
#             plt.savefig('splits.'+str(i)+'.png')

#         figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,ndim=6)
#         dm.data_dm().plot(lambda x: outliers,
#             color='b',alpha=0.5,s=20,ndim=6,figax=figax)
#         plt.savefig('temp.png')
#         figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
#         dm.data_dm().plot(lambda x: outliers,
#             color='b',alpha=0.5,s=20,ndim=6,figax=figax)
#         plt.savefig('temp2.png')


#         # plt.clf()
#         # figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         # test_data.plot(lambda x: w, label='0.2',color='b',alpha=1,s=10,figax=figax)
#         # plt.show()

#     train_dist = sklearn.metrics.pairwise_distances(dm.data_dm().x,dm.data_dm().x)   
#     outlier_distances = train_dist[numpy.outer(outliers,outliers)]
#     outlier_distances = outlier_distances[outlier_distances !=0]
#     outlier_distances = numpy.sort(outlier_distances)
#     norm = outlier_distances[len(outlier_distances)*.05]

#     numpy.fill_diagonal(train_dist,train_dist.max()) #numpy.finfo('d').max)
#     train_min_dist = numpy.min(train_dist,axis=0)
#     closer  = train_min_dist < numpy.sort(train_min_dist)[train_data.ndata()*.95]
#     weight = numpy.exp(-(train_dist[outliers,:]/norm)**2).sum(axis=0)
#     if doplot:    
#         cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
#         cval=cm.to_rgba(weight)
#         figax= train_data.plot(c=cval,alpha=0.1,s=20,cmap=cm)

#     weight_sort=numpy.sort(weight)

#     mn=[]
#     sd=[]
#     fracs=numpy.arange(0.01,0.5,0.05)
#     for frac in fracs:
#         w=weight < weight_sort[frac*train_data.ndata()]
#         w=numpy.logical_and(w,closer)
#         mn.append(train_data.y[w].mean())
#         sd.append(train_data.y[w].std())
#     plt.scatter(fracs,mn,c='b',label='mean')
#     plt.scatter(fracs,sd,c='r',label='std')
#     plt.show()

#     wfe
    # get the subset of bad ones

    # train_min_dist=numpy.min(train_dist,axis=0)
    # train_sort = numpy.argsort(train_min_dist)
    # train_sort = train_sort[0:prunefrac * len(train_sort)]
    # train_cut =  train_sort[-1]
    # test_dist = sklearn.metrics.pairwise_distances(dmsys.dmdata.x,test_data_dm.x)
    # test_min_dist = numpy.min(test_dist,axis=0)
    # test_sort =  numpy.sort(test_min_dist)
    # w = test_min_dist < test_sort[len(test_sort)*prunefrac]
#     plt.clf()


#    plt.savefig('temp1.png')


    # wef
# #    plt.savefig('temp2.png')
    # figax= test_data_dm.plot(color='r',alpha=0.1,s=10)
    # test_data_dm.plot(lambda x: numpy.abs(test_data_dm.y) >x0[0], color='b',alpha=0.5,s=20,figax=figax)
    # plt.show()
    #    plt.savefig('temp3.png')



#     wefe

