#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'

import os
import pickle
#import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
from argparse import ArgumentParser
import diffuse
import scipy
from scipy.stats import norm
import sklearn
from sklearn.cross_validation import cross_val_score
import sklearn.ensemble
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import sklearn.metrics.pairwise
 
import copy
mpl.rc('figure', figsize=(11,11))

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

"""
The parameters that are to be optimized
"""
parameter_names =['bias_threshold','eps_val']

def split(x):
    #(0.8, 0.33), (1.1, 0.42)
    return x[:,1] < 0.33 + (0.42-0.33)/(1.1-0.8)*(x[:,0]-0.8)

from matplotlib.legend_handler import HandlerNpoints

class Plots:
    @staticmethod
    def x(x,xlabel=None, figax='default',nsig=None,**kwargs):

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


class Data:

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

    def __init__(self, x, y, z, xlabel = None, ylabel=None, zlabel=None):
        self.x=x
        self.y=y
        self.z=z
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.zlabel=zlabel

    def __getitem__(self , index):
        return Data(self.x[index],self.y[index],self.z[index])

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
#    Plots.x(X_train,good=numpy.abs(y_train) <0.01)
#    wfe
    return Data(X_train, y_train, z_train,xlabel=['g-r','r-i','i-z','i'], ylabel='bias',zlabel='photo-z'), Data(X_test,  y_test, z_test,xlabel=['g-r','r-i','i-z','i'], ylabel='bias',zlabel='photo-z')


class DiffusionMap:

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

    def __init__(self, data, par, label=None):
       self.data = data
       self.par = par  #for the moment eps_val
       self.key=label
       self.weight= numpy.array([2,2,1,1])
       self.nvar = 6


    def make_map(self):

        """ Method that calculates the diffusion map.  The result is
        stored internally.
        """

        kwargs=dict()
        kwargs['eps_val'] = self.par.item()
        kwargs['t']=1
        kwargs['delta']=1e-8
        kwargs['var']=0.95

        self.dmap = diffuse.diffuse(self.data.x*self.weight,**kwargs)

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

        return diffuse.nystrom(self.dmap, self.data.x*self.weight, x*self.weight)[:,0:self.nvar]

    def transform_full(self, x):

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

    def internal_coordinates(self):
        return self.dmap['X'][:,0:self.nvar]

    def internal_coordinates_full(self):
        return self.dmap['X']

# New coordinate system is based on a set of diffusion maps
class DMSystem:
    """ Class that manages the new coordinate system we want.
    The new system is the union of several diffusion maps.

    Parameters
    ----------
    data : Data
       Data that enters into the diffusion maps
    """
    @staticmethod
    def hasTrainingAnalog(dmcoords):
        ans = numpy.empty(dmcoords.shape[0],dtype='bool')

        for i in xrange(dmcoords.shape[0]):
            ans[i] = not numpy.isnan(dmcoords[i,:]).any()

        return ans


    def __init__(self, data, config = 'default'):
        self.data = data 
        
        self.state = None

        # if config == 'default':
        #     self.config = self._onebad
        # else:
        #     self.config = self._redshift_bins

    def config(self, good_inds, bad_inds, par):
        self.dm=[]
        # key = dict()
        # newmap=DiffusionMap(self.data[bad_inds],par[1],key)
        # key['dm']=newmap
        # key['bias']=False
        # key['zbin']=0
        # key['train_inds'] = bad_inds
        # self.dm.append(key)

        key = dict()
        newmap=DiffusionMap(self.data[good_inds],par[1],key)
        key['dm']=newmap
        key['bias']=True
        key['zbin']=0
        key['train_inds'] = good_inds
        self.dm.append(key)

        self.nvar = 0
        for dm in self.dm:
            self.nvar = self.nvar+dm['dm'].nvar

    def create_dm(self, par):

        if numpy.array_equal(self.state,par):
            return
        bias = self.data.y

    # Split into "good" and "bad" samples
        good_inds = numpy.abs(bias) <= par[0]
        bad_inds = numpy.logical_not(good_inds)
        self.good_inds= good_inds
        self.bad_inds=bad_inds
        self.config(good_inds, bad_inds, par)



        self.state = numpy.array(par)

        train_coordinates = numpy.zeros((len(self.data.x),  self.nvar))

        # put the new coordinates into a data
        ncoord=0
        for dm in self.dm:
            dm['dm'].make_map()
            train_coordinates[dm['train_inds'],ncoord:ncoord+dm['dm'].nvar]= \
                dm['dm'].internal_coordinates()
            train_coordinates[numpy.logical_not(dm['train_inds']),ncoord:ncoord+dm['dm'].nvar]= \
                dm['dm'].transform(self.data.x[numpy.logical_not(dm['train_inds'])])
            ncoord=ncoord+dm['dm'].nvar

        # renormalize the coordinates to be sane
        self.mns=[]
#        self.sigs=[]

        for index in xrange(ncoord):
            xso=numpy.sort(train_coordinates[:,index])
            l= len(xso)
            xso=xso[l*.2:l*.8]
            xmn = xso[len(xso)/2]
#            xsig = xso.std()
            train_coordinates[:,index]=(train_coordinates[:,index]-xmn)#/xsig
            self.mns.append(xmn)
#            self.sigs.append(xsig)
        self.mns=numpy.array(self.mns)

        xso = numpy.sort(train_coordinates)
        l=len(xso)
        xso=xso[l*.2:l*.8]
        self.sig = xso.std()
        train_coordinates = train_coordinates/self.sig

        #self.sigs=numpy.array(self.sigs)
        self.dmdata = Data(train_coordinates,self.data.y,self.data.z,xlabel=[str(i) for i in xrange(self.nvar)])

    def coordinates(self, x, par):

        # if the current state of the diffusion maps is not equal
        # to what is requested make them
        self.create_dm(par)

        coords = numpy.empty((len(x),0))
        for dm in self.dm:
            coords=numpy.append(coords, dm['dm'].transform(x),axis=1)

        for index in xrange(len(self.mns)):
            coords[:,index]=(coords[:,index]-self.mns[index])/self.sig
        return coords



# import sklearn.ensemble
# class MyRegressor(sklearn.ensemble.forest.ForestRegressor):
#     """A random forest regressor.

#     A random forest is a meta estimator that fits a number of classifying
#     decision trees on various sub-samples of the dataset and use averaging
#     to improve the predictive accuracy and control over-fitting.

#     Parameters
#     ----------
#     n_estimators : integer, optional (default=10)
#         The number of trees in the forest.

#     criterion : string, optional (default="mse")
#         The function to measure the quality of a split. The only supported
#         criterion is "mse" for the mean squared error.
#         Note: this parameter is tree-specific.

#     max_features : int, float, string or None, optional (default="auto")
#         The number of features to consider when looking for the best split:

#         - If int, then consider `max_features` features at each split.
#         - If float, then `max_features` is a percentage and
#           `int(max_features * n_features)` features are considered at each
#           split.
#         - If "auto", then `max_features=n_features`.
#         - If "sqrt", then `max_features=sqrt(n_features)`.
#         - If "log2", then `max_features=log2(n_features)`.
#         - If None, then `max_features=n_features`.

#         Note: the search for a split does not stop until at least one
#         valid partition of the node samples is found, even if it requires to
#         effectively inspect more than ``max_features`` features.
#         Note: this parameter is tree-specific.

#     max_depth : integer or None, optional (default=None)
#         The maximum depth of the tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.
#         Ignored if ``max_samples_leaf`` is not None.
#         Note: this parameter is tree-specific.

#     min_samples_split : integer, optional (default=2)
#         The minimum number of samples required to split an internal node.
#         Note: this parameter is tree-specific.

#     min_samples_leaf : integer, optional (default=1)
#         The minimum number of samples in newly created leaves.  A split is
#         discarded if after the split, one of the leaves would contain less then
#         ``min_samples_leaf`` samples.
#         Note: this parameter is tree-specific.

#     max_leaf_nodes : int or None, optional (default=None)
#         Grow trees with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.
#         If not None then ``max_depth`` will be ignored.
#         Note: this parameter is tree-specific.

#     bootstrap : boolean, optional (default=True)
#         Whether bootstrap samples are used when building trees.

#     oob_score : bool
#         whether to use out-of-bag samples to estimate
#         the generalization error.

#     n_jobs : integer, optional (default=1)
#         The number of jobs to run in parallel for both `fit` and `predict`.
#         If -1, then the number of jobs is set to the number of cores.

#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.

#     verbose : int, optional (default=0)
#         Controls the verbosity of the tree building process.

#     Attributes
#     ----------
#     `estimators_`: list of DecisionTreeRegressor
#         The collection of fitted sub-estimators.

#     `feature_importances_` : array of shape = [n_features]
#         The feature importances (the higher, the more important the feature).

#     `oob_score_` : float
#         Score of the training dataset obtained using an out-of-bag estimate.

#     `oob_prediction_` : array of shape = [n_samples]
#         Prediction computed with out-of-bag estimate on the training set.

#     References
#     ----------

#     .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

#     See also
#     --------
#     DecisionTreeRegressor, ExtraTreesRegressor
#     """
#     def __init__(self, eval_frac=0.1,
#                  n_estimators=10,
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0,
#                  min_density=None,
#                  compute_importances=None):
#         super(MyRegressor, self).__init__(
#             base_estimator=sklearn.ensemble.forest.DecisionTreeRegressor(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "max_features",
#                               "max_leaf_nodes", "random_state"),
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose)

#         self.eval_frac=eval_frac
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes

#         if min_density is not None:
#             warn("The min_density parameter is deprecated as of version 0.14 "
#                  "and will be removed in 0.16.", DeprecationWarning)

#         if compute_importances is not None:
#             warn("Setting compute_importances is no longer required as "
#                  "version 0.14. Variable importances are now computed on the "
#                  "fly when accessing the feature_importances_ attribute. "
#                  "This parameter will be removed in 0.16.",
#                  DeprecationWarning)

#     def score(self,X, y, sample_weight=None):
#         return (self.score_info(X, y, sample_weight=sample_weight))[0]
# #         y_predict, y_var =self.predict(X)
# #         delta_y=y_predict-y
# #         sin = numpy.argsort(y_var)
# #         sin=sin[0:len(sin)*self.eval_frac]
# #         delta_y=delta_y[sin]
# #         return numpy.exp(-delta_y.std())

#     def score_info(self,X, y, sample_weight=None):
#         y_predict, y_var =self.predict(X)
#         delta_y=y_predict-y
#         sin = numpy.argsort(y_var)
#         sin=sin[0:len(sin)*self.eval_frac]
#         delta_y=delta_y[sin]
#         return numpy.exp(-delta_y.std()),sin

class Objective(object):
    """docstring for Objective"""
    def __init__(self, x, y):
        super(Objective, self).__init__()
        self.x = x
        self.y = y
        self.clf=MyRegressor(0.1)
        self.frac_include  =  0.01*1.5**numpy.arange(12) #12

    def set_regressor(self,par):
        kwargs=dict()
        kwargs['n_estimators']=int(par[0])
        kwargs['max_features']=int(par[1])
        kwargs['min_samples_leaf']=int(par[2])
        kwargs['random_state']=int(par[3])
        self.clf.set_params(**kwargs)

    def eval(self,par):
        # kwargs=dict()
        # kwargs['n_estimators']=int(par[0])
        # kwargs['max_features']=int(par[1])
        # kwargs['min_samples_leaf']=int(par[2])
        # kwargs['random_state']=int(par[3])
        # clf=self.clf.set_params(**kwargs)
        self.set_regressor(par)
        scores = cross_val_score(self.clf, self.x, self.y, cv=5)
        score = -numpy.log(scores.mean())
        return score

    def optimize(self,**kwargs):
    # the order of the variables
    # n_estimators
    # max_features
    # min_samples_leaf
    # seed
        x0s = []
        fvals = []
        for frac in self.frac_include:
            self.clf.eval_frac=frac
            x0, fval, grid, jout = scipy.optimize.brute(self.eval,
                finish=False,full_output=True, **kwargs)
            x0s.append(x0)
            fvals.append(fval)
        self.x0s=numpy.array(x0s)
        self.fvals=numpy.array(fvals)
        return x0s, fvals

    def plot_scatter(self,ax,**kwargs):
        ax.scatter(self.frac_include,self.fvals,**kwargs)

    def plot_scatter_external(self,ax,x,y,**kwargs):
        test_score_color=[]
        for ind in xrange(len(self.frac_include)):
            self.set_regressor(self.x0s[ind])
            self.clf.eval_frac=self.frac_include[ind]
            self.clf.fit(self.x,self.y)
            predict_color, sin = self.clf.score_info(x,y)
            print len(sin)
            test_score_color.append(predict_color)
        test_score_color=numpy.array(test_score_color)
        test_score_color = -numpy.log(test_score_color)
        ax.scatter(self.frac_include,test_score_color,**kwargs)
    # @staticmethod
    # def crap(y, y_predict, y_var,eval_frac):
    #     delta_y=y_predict-y
    #     sin = numpy.argsort(y_var)
    #     sin=sin[0:len(sin)*eval_frac]
    #     delta_y=delta_y[sin]
    #     return delta_y.mean(),delta_y.std()

def passlimit(output,o,frac):
    w  = output.test_min_dist < output.train_cut
    temp = numpy.sort(o.dy2[w])
    limit = temp[frac*len(w)]
    w2 = o.dy2 < limit
    w3 = numpy.logical_and(w,w2)
    return w3

def meanuncertainties(test_data,output):
    mns=[]
    dmns=[]
    sigs=[]
    dsigs=[]
    frac_include  =  0.01*1.5**numpy.arange(12)
    frac_include = numpy.arange(0.05,.9,0.05)
    # w  = output.test_min_dist < output.train_cut

    allmn=[]
    allsd=[]
    for frac in frac_include:
        mn=[]
        sd=[]
#        print frac,
        for o in output.outputs:
            w3=passlimit(output,o,frac)

#             temp = numpy.sort(o.dy2[w])
#             limit = temp[frac*len(w)]
# #            print limit,
#             w2 = o.dy2 < limit
#             w3 = numpy.logical_and(w,w2)
            d = o.y-test_data.y
            d = d[w3]
            mn.append(d.mean())
            sd.append(d.std())
 #       print
        mn=numpy.array(mn)
        sd=numpy.array(sd)
        allmn.append(mn)
        allsd.append(sd)
        mns.append(mn.mean())
        dmns.append(mn.std())
        sigs.append(sd.mean())
        dsigs.append(sd.std())
        # plt.hist(sd)
        # plt.show()
    mns=numpy.array(mns)
    dmns=numpy.array(dmns)
    sigs=numpy.array(sigs)
    dsigs=numpy.array(dsigs)
    allmn=numpy.array(allmn)
    allsd=numpy.array(allsd)
    return frac_include, mns,dmns, sigs, dsigs, allmn,allsd

    # # construct difference in y
    # delta_y = []
    # for o in output.outputs:
    #     d = o.y-test_data.y


    #     temp = numpy.sort(o.dy2[w])
    #     w2 = o.dy2 < temp
    #     w3 = numpy.logical_and(w,w2)
    #     delta_y.append(delta_y[w3])




    # dum=[]
    # mns=[]
    # for frac in frac_include:
    #     dum2=[]
    #     mn2=[]
    #     for color_out in output.outputs:
    #         w= output.test_min_dist<output.train_cut
    #         mn, sig = Objective.crap(test_data.y[w], color_out.y[w],color_out.dy2[w],frac)
    #         dum2.append(sig)
    #         mn2.append(mn)
    #     dum2=numpy.array(dum2)
    #     dum.append(dum2)
    #     mn2=numpy.array(mn2)
    #     mns.append(mn2)

    # dum=numpy.array(dum)
    # mns=numpy.array(mns)
    # means = dum.mean(axis=1)
    # stds = dum.std(axis=1)

    # return frac_include,mns.mean(axis=1),mns.std(axis=1),dum.mean(axis=1),dum.std(axis=1)

def ok(x):
    w=x[:,5]>0.001
    w=x[:,5]<0
    w=numpy.logical_and(w,x[:,4]>5e-3)
    w=x[:,3]<-0.1
    w=x[:,5]>0
    w=numpy.logical_and(w,x[:,3]>0.01)
    w=numpy.logical_and(w,x[:,4]>2e-4)
    return w

if __name__ == '__main__':

    doplot = False
    parser = ArgumentParser()
    parser.add_argument('test_size', nargs='?',default=0.1)
    parser.add_argument('seed', nargs='?',default=9)
    
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    x0 = numpy.array([0.05,0.0025*2.5])
    x0 = numpy.array([1,0.0025])

    prob=0.9

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)

    import os.path
    if os.path.isfile('dmsys.pkl'):
        #print 'get pickle'
        pklfile=open('dmsys.pkl','r')
        dmsys=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        dmsys= DMSystem(train_data)

        # for x01 in [0.0025]:
        #     x0[1]=x01
        dmsys.create_dm(x0)

        #print 'make pickle'
        pklfile=open('dmsys.pkl','w')
        pickle.dump(dmsys,pklfile)
    pklfile.close()

    filename='dmsys2.pkl'
    if os.path.isfile(filename):
        #print 'get pickle'
        pklfile=open(filename,'r')
        test_data_dm,test_data_dm_full=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z,
            xlabel=[str(i) for i in xrange(dmsys.nvar)])
        test_data_dm_full  = Data(dmsys.dm[0]['dm'].transform_full(test_data.x),test_data.y,test_data.z)
        #print 'make pickle'
        pklfile=open(filename,'w')
        pickle.dump([test_data_dm,test_data_dm_full],pklfile)
    pklfile.close()

    prunefrac=0.95
    # train_dist = sklearn.metrics.pairwise_distances(dmsys.dmdata.x,dmsys.dmdata.x)
    # numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
    # train_min_dist=numpy.min(train_dist,axis=0)
    # train_sort = numpy.argsort(train_min_dist)
    # train_sort = train_sort[0:prunefrac * len(train_sort)]
    # train_cut =  train_sort[-1]
    # test_dist = sklearn.metrics.pairwise_distances(dmsys.dmdata.x,test_data_dm.x)
    # test_min_dist = numpy.min(test_dist,axis=0)
    # test_sort =  numpy.sort(test_min_dist)
    # w = test_min_dist < test_sort[len(test_sort)*prunefrac]
#     plt.clf()
    # figax= train_data.plot(color='r',alpha=0.1,s=10)
    # train_data.plot(lambda x: numpy.abs(train_data.y) >0.1, color='b',alpha=0.5,s=20,figax=figax)
    # plt.show()

#    plt.savefig('temp1.png')
    # figax= dmsys.dmdata.plot(color='r',alpha=0.1,s=10)
    # dmsys.dmdata.plot(lambda x: numpy.abs(test_data_dm.y) > 0.05, color='b',alpha=0.5,s=20,figax=figax)
    # plt.savefig('temp.png')
    # figax= dmsys.dmdata.plot(color='r',alpha=0.1,s=10,nsig=20)
    # dmsys.dmdata.plot(lambda x: numpy.abs(test_data_dm.y) > 0.05, color='b',alpha=0.5,s=20,figax=figax)
    # plt.savefig('temp2.png')

    # wef
# #    plt.savefig('temp2.png')
    # figax= test_data_dm.plot(color='r',alpha=0.1,s=10)
    # test_data_dm.plot(lambda x: numpy.abs(test_data_dm.y) >x0[0], color='b',alpha=0.5,s=20,figax=figax)
    # plt.show()
    #    plt.savefig('temp3.png')



#     wefe
    import matplotlib.cm
    import matplotlib.colors
    ## plots that show dm x in color space
    if doplot:
        for i in xrange(6):
            crap = numpy.sort(dmsys.dmdata.x[:,i])
            crap= crap[len(crap)*.1:len(crap)*.9]
            sig = crap.std()
            cm=matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=crap[len(crap)/2]-5*sig,
                    vmax=crap[len(crap)/2]+5*sig),cmap='Spectral')
            cval=cm.to_rgba(dmsys.dmdata.x[:,i])
            figax= train_data.plot(c=cval,alpha=0.3,s=20,cmap=cm)
            figax[0].suptitle(str(i))
            plt.savefig('splits.'+str(i)+'.png')
        wef
        print test_data.y.mean(), test_data.y.std()
        w=ok(test_data_dm.x)
        print test_data.y[w].mean(), test_data.y[w].std()
        print len(test_data.y),w.sum()
        plt.clf()
        figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
        test_data.plot(lambda x: w, label='0.2',color='b',alpha=1,s=10,figax=figax)
        plt.show()
        wefe

    #get distances
    
    catastrophic = numpy.abs(test_data.y)>0.05
    dist = sklearn.metrics.pairwise_distances(dmsys.dm[0]['dm'].internal_coordinates_full[:,catastrophic],
        test_data_dm_full.x)

    prunefrac=0.95

 
    class OneOutput(object):
        """docstring for OneOutput"""
        def __init__(self, feature_importances_,y,dy2):
            super(OneOutput, self).__init__()
            self.feature_importances_ = feature_importances_
            self.y=y
            self.dy2=dy2
            
    class Output(object):
        """docstring for Output"""
        def __init__(self,test_min_dist,train_cut, outputs):
            super(Output, self).__init__()
            self.test_min_dist=test_min_dist
            self.train_cut=train_cut
            self.outputs=outputs
    filename='output.pkl'

    if os.path.isfile(filename):
        #print 'get pickle'
        pklfile=open(filename,'r')
        coloroutput,dmoutput=pickle.load(pklfile)
    else:
        nrealize=100
        train_dist = sklearn.metrics.pairwise_distances(train_data.x,train_data.x)
        numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
        train_min_dist=numpy.min(train_dist,axis=0)
        train_sort = numpy.argsort(train_min_dist)
        train_sort = train_sort[0:prunefrac * len(train_sort)]
        train_cut =  train_min_dist[train_sort[-1]]
        test_dist = sklearn.metrics.pairwise_distances(train_data.x[train_sort],test_data.x)
        test_min_dist = numpy.min(test_dist,axis=0)

        clf = sklearn.ensemble.forest.RandomForestRegressor(n_estimators=100,random_state=12)
        color_outs=[]
        for rs in xrange(nrealize):
            clf.set_params(random_state=rs)
            clf.fit(train_data.x[train_sort],train_data.y[train_sort])
            y,dy2 = clf.predict(test_data.x)
            color_outs.append(OneOutput(clf.feature_importances_,y,dy2))
        coloroutput=Output(test_min_dist,train_cut,color_outs)

        train_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,0:4],dmsys.dmdata.x[:,0:4])**2+
            sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,4:],dmsys.dmdata.x[:,4:])**2)
        numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
        train_min_dist=numpy.min(train_dist,axis=0)
        train_sort = numpy.argsort(train_min_dist)
        train_sort = train_sort[0:prunefrac * len(train_sort)]
        train_cut =  train_min_dist[train_sort[-1]]
        test_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,0:4],
            test_data_dm.x[:,0:4])**2+sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,4:]
            ,test_data_dm.x[:,4:])**2)
        test_min_dist = numpy.min(test_dist,axis=0)

        dm_outs=[]
        for rs in xrange(nrealize):
            clf.set_params(random_state=rs)
            clf.fit(dmsys.dmdata.x[train_sort],dmsys.dmdata.y[train_sort])
            y,dy2 = clf.predict(test_data_dm.x)
            dm_outs.append(OneOutput(clf.feature_importances_,y,dy2))
        dmoutput=Output(test_min_dist,train_cut,dm_outs)

        pklfile=open(filename,'w')
        pickle.dump([coloroutput,dmoutput],pklfile)
    pklfile.close()


    # w  = dmoutput.test_min_dist < dmoutput.train_cut
    # plt.scatter(dmoutput.test_min_dist, dmoutput.outputs[0].y-test_data_dm.y)
    # plt.xlim(0,dmoutput.train_cut*2)
    # plt.show()

    w  = coloroutput.test_min_dist < coloroutput.train_cut
    plt.scatter(coloroutput.test_min_dist, coloroutput.outputs[0].y-test_data_dm.y)
    plt.xlim(0,coloroutput.train_cut*2)
    plt.show()

    plt.scatter(dmoutput.outputs[0].dy2[w],dmoutput.outputs[0].y[w]-test_data_dm.y[w])
    plt.show()

    wef

    if doplot:      
        plt.clf()
        figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
        test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
        test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
            figax=figax)
        plt.show()

        plt.clf()
        figax= test_data_dm.plot(label='0.3',color='k',alpha=0.1,s=10,nsig=4)
        test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2),nsig=4, label='0.2',color='b',alpha=1,s=10,figax=figax)
        test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1),nsig=4 , label='0.1',color='r',alpha=1,s=10,
            figax=figax)
        plt.show()

        plt.clf()
        figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
        test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
        test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
            figax=figax)
        plt.show()

        figax= test_data_dm.plot(label='0.3',color='k',nsig=4,alpha=0.1,s=10)
        test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',nsig=4,alpha=1,s=10,figax=figax)
        test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',nsig=4,alpha=1,s=10,
            figax=figax)
        plt.show()

    c1,c2,c3,c4,c5, c6, c7 = meanuncertainties(test_data,coloroutput)
    d1,d2,d3,d4,d5,d6,d7= meanuncertainties(test_data_dm,dmoutput)
    # ind=0
    # plt.clf()
    # fig=plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(c6[ind,:],label='c')
    # ax.plot(d6[ind,:],label='d')
    # plt.legend()
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(c1,c2,yerr=c3,label='color',marker='o',color='b')
    ax.errorbar(d1,d2,yerr=d3,label='dm',marker='o',color='r')
    plt.legend()
    plt.show()
    fwe
    # for a,b,c, d, e in zip(frac_include,mns.mean(axis=1),mns.std(axis=1),means,stds):
    #     print "{:5.3f} {:7.4f} {:6.4f} {:6.4f} {:6.4f}".format(a,b,c, d, e)



#     wefew
#     dum=[]
#     train_dist = sklearn.metrics.pairwise_distances(train_data.x,train_data.x)
#     numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
#     train_min_dist=numpy.min(train_dist,axis=0)
#     train_sort = numpy.argsort(train_min_dist)
#     train_sort = train_sort[0:prunefrac * len(train_sort)]
#     train_cut =  train_sort[-1]
#     test_dist = sklearn.metrics.pairwise_distances(train_data.x[train_sort],test_data.x)
#     test_min_dist = numpy.min(test_dist,axis=0)

#     clf = sklearn.ensemble.forest.RandomForestRegressor(n_estimators=100,random_state=12)
#     for rs in xrange(10):
#         clf.set_params(random_state=rs)
#         clf.fit(train_data.x[train_sort],train_data.y[train_sort])

# #        y,dy = clf.predict(test_data.x[test_min_dist<train_cut])
#         y,dy = clf.predict(test_data.x)
#         dum2=[]
#         w= test_min_dist<train_cut
#         for frac in frac_include:
#             dum2.append(Objective.crap(test_data.y[w],y[w],
#                 dy[w],frac)[1])
#             # dum2.append(Objective.crap(test_data.y[test_min_dist<train_cut],y,
#             #     dy,frac)[1])
#         dum2=numpy.array(dum2)
#         dum.append(dum2)
#     dum=numpy.array(dum)
#     print dum
#     wefwe
#     means = dum.mean(axis=0)
#     stds = dum.std(axis=0)
#     for a,b,c in zip(frac_include,means,stds):
#         print a,b,c

#     wefe

    # frac_include  =  0.01*1.5**numpy.arange(12)
    
    # train_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,0:4],dmsys.dmdata.x[:,0:4])**2+
    #     sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,4:],dmsys.dmdata.x[:,4:])**2)
    # numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
    # train_min_dist=numpy.min(train_dist,axis=0)
    # train_sort = numpy.argsort(train_min_dist)
    # train_sort = train_sort[0:prunefrac * len(train_sort)]
    # train_cut =  train_sort[-1]
    # test_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,0:4],test_data_dm.x[:,0:4])**2+
    #     sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,4:],test_data_dm.x[:,4:])**2)
    # test_min_dist = numpy.min(test_dist,axis=0)
    # dum=[]
    # clf = sklearn.ensemble.forest.RandomForestRegressor(n_estimators=50,random_state=12)
    # for rs in xrange(10):
    #     clf.fit(dmsys.dmdata.x[train_sort],dmsys.dmdata.y[train_sort])
    #     y,dy = clf.predict(test_data_dm.x[test_min_dist<train_cut])
    #     print Objective.crap(test_data_dm.y[test_min_dist<train_cut],y,dy,0.01)
        
    #     # dum2=[]
    #     # for frac in frac_include:
    #     #     dum2.append(Objective.crap(test_data_dm.y[test_min_dist<train_cut],y,dy,frac)[1])
    #     #     dum2=numpy.array(dum2)
    #     # print frac,dum2.mean(),dum2.std()
    # wefwf
    # dum.append(dum2)
    # dum=numpy.array(dum)

    # fwe
    if doplot:
        for index in xrange(8):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(dmsys.dmdata.x[:,index],dmsys.dmdata.y)
            ax.set_xlim((-20,20))
        plt.show()
    import matplotlib.collections
    import matplotlib.lines
    import matplotlib.patches
    if doplot:
        ## stuff to force the legend to have alpha=1
        labs=['positive bias','negative bias','no bias']
        # lines = [matplotlib.lines.Line2D([],[],color='r',marker='.',linestyle='None'),
        # matplotlib.lines.Line2D([],[],color='b',marker='.',linestyle='None')
        # ,matplotlib.lines.Line2D([],[],color='k',marker='.',linestyle='None')]
        lines = [matplotlib.patches.Circle([], color='r'),matplotlib.patches.Circle([], color='b')
            ,matplotlib.patches.Circle([], color='k')]

        plt.clf()
       # w = numpy.logical_and(split(train_data.x), numpy.abs(train_data.y) > x0[0])                    
        figax= train_data.plot(lambda x: train_data.y > x0[0] , label='positive bias',color='r',alpha=0.02)
        train_data.plot(lambda x: train_data.y < -x0[0] , label='negative bias',color='b',figax=figax,alpha=0.02)
        train_data.plot(lambda x: numpy.abs(train_data.y) <= x0[0] , label='no bias',color='k',
            figax=figax,alpha=0.02)

        for ax in figax[1]:
            for a in ax:
                a.legend(lines, labs,prop={'size':6})

        plt.savefig('../results/colorspace.png')

        plt.clf()
        figax = dmsys.dmdata.plot(lambda x: dmsys.dmdata.y > x0[0] ,
            label='positive bias',color='r',nsig=4,alpha=0.01)
        dmsys.dmdata.plot(lambda x: dmsys.dmdata.y < -x0[0] ,
            label='negative bias',color='b',figax=figax,nsig=4,alpha=0.01)
        dmsys.dmdata.plot(lambda x: numpy.abs(dmsys.dmdata.y) <= x0[0] ,
            label='no bias',color='k',figax=figax,nsig=4,alpha=0.01)

        for ax in figax[1]:
            for a in ax:
                a.legend(lines, labs,prop={'size':3})
        plt.savefig('../results/dmspace.png')
     

    filename = 'optimum.pkl'

    if os.path.isfile(filename):
        pklfile=open(filename,'r')
        (objective_dm, objective_color)=pickle.load(pklfile)
    else:
        objective_dm = Objective(dmsys.dmdata.x,dmsys.dmdata.y)
        objective_color = Objective(train_data.x,train_data.y)
        objective_dm.optimize(ranges=(slice(10,60,5),
            slice(3,objective_dm.x.shape[1]+1,1),slice(1,10,1),slice(0,10,1)))
        objective_color.optimize(ranges=(slice(10,60,5),
            slice(3,objective_color.x.shape[1]+1,1),slice(1,10,1),slice(0,10,1)))
 
        pklfile=open(filename,'w')
        pickle.dump((objective_dm, objective_color),pklfile)
    pklfile.close()


    # clf = objective_dm.clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)
    # test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
    # y_pred, y_var = clf.predict(test_data_dm.x)
    # y_sig=numpy.sqrt(y_var)
    # delta_y = y_pred-test_data.y
    # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)
    # sin = numpy.argsort(y_sig)
    # delta_y=delta_y[sin]
    # y_sig=y_sig[sin]
    # std=[]
    # predind = numpy.arange(10,len(delta_y),10)
    # for i in xrange(len(predind)-1):
    #     std.append(delta_y[predind[i]:predind[i+1]].std())
    # std=numpy.array(std)
    # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)
    # ax1.legend()
    # ax2.legend()
    # plt.show()
    

  # end old code
    test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
    dists=numpy.sqrt(numpy.dot(dmsys.dmdata.x[:,0:4],numpy.transpose(test_data_dm.x[:,0:4]))**2 +
        numpy.dot(dmsys.dmdata.x[:,4:],numpy.transpose(test_data_dm.x[:,4:]))**2)
    dists = numpy.max(dists,axis=0)
    w=dists < 1e15

    # if True:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     objective_dm.plot_scatter_external(ax,test_data_dm.x[w],test_data_dm.y[w],label='dm',color='b')
    #     objective_color.plot_scatter_external(ax,test_data.x[w],test_data.y[w],label='color',color='r')
    #     objective_dm.plot_scatter_external(ax,dmsys.dmdata.x[w],dmsys.dmdata.y[w],label='dm',color='b',marker='x')
    #     objective_color.plot_scatter_external(ax,train_data.x[w],train_data.y[w],label='color',color='r',marker='x')

    #     ax.legend()
    #     plt.show()
    #     wefwe

    if True:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        ax.scatter(objective_dm.frac_include,objective_dm.fvals,label='dm',color='b')
#        ax.scatter(objective_color.frac_include,objective_color.fvals,label='color',color='r')
        objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b',marker='x')
        objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r',marker='x')
        ax.legend()
        plt.savefig('../results/sigmas.png')
        wefwe


    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        objective_dm.plot_scatter(ax,label='Train dm',color='b')
        objective_color.plot_scatter(ax,label='Train color',color='r')
        ax.legend()

    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b')
        objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r')


        ax.legend()
        plt.show()

    # clf = clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)

    # y_pred, y_var = clf.predict(dmsys.dmdata.x)
    # y_sig=numpy.sqrt(y_var)

    # delta_y = y_pred-dmsys.dmdata.y

    # fig1 = plt.figure()

    # ax1 = fig1.add_subplot(111)
    # ax1.scatter(y_sig, delta_y, label='train')

    # sin = numpy.argsort(y_sig)
    # delta_y=delta_y[sin]
    # y_sig=y_sig[sin]
    # std=[]
    # predind = numpy.arange(10,len(delta_y),10)
    # for i in xrange(len(predind)-1):
    #     std.append(delta_y[predind[i]:predind[i+1]].std())
    # std=numpy.array(std)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.scatter(y_sig[predind[1:]],std,label='train')


    # 
    # y_pred, y_var = clf.predict(test_data_dm.x)
    # y_sig=numpy.sqrt(y_var)

    # delta_y = y_pred-test_data.y

    # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)

    # sin = numpy.argsort(y_sig)
    # delta_y=delta_y[sin]
    # y_sig=y_sig[sin]
    # std=[]
    # predind = numpy.arange(10,len(delta_y),10)
    # for i in xrange(len(predind)-1):
    #     std.append(delta_y[predind[i]:predind[i+1]].std())
    # std=numpy.array(std)
    # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)

    # ax1.legend()
    # ax2.legend()

    # plt.show()
