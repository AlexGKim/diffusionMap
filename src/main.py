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
import diffuse
import copy
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
        self.neigen = self.dmap['neigen']

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


# # New coordinate system is based on a set of diffusion maps
# class DMSystem:
#     """ Class that manages the new coordinate system we want.
#     The new system is the union of several diffusion maps.

#     Parameters
#     ----------
#     data : Data
#        Data that enters into the diffusion maps
#     """
#     @staticmethod
#     def hasTrainingAnalog(dmcoords):
#         ans = numpy.empty(dmcoords.shape[0],dtype='bool')

#         for i in xrange(dmcoords.shape[0]):
#             ans[i] = not numpy.isnan(dmcoords[i,:]).any()

#         return ans


#     def __init__(self, data, config = 'default'):
#         self.data = data 
        
#         self.state = None

#         # if config == 'default':
#         #     self.config = self._onebad
#         # else:
#         #     self.config = self._redshift_bins

#     def config(self, good_inds, bad_inds, par):
#         self.dm=[]
#         # key = dict()
#         # newmap=DiffusionMap(self.data[bad_inds],par[1],key)
#         # key['dm']=newmap
#         # key['bias']=False
#         # key['zbin']=0
#         # key['train_inds'] = bad_inds
#         # self.dm.append(key)

#         key = dict()
#         newmap=DiffusionMap(self.data[good_inds],par[1],key)
#         key['dm']=newmap
#         key['bias']=True
#         key['zbin']=0
#         key['train_inds'] = good_inds
#         self.dm.append(key)

#         self.nvar = 0
#         for dm in self.dm:
#             self.nvar = self.nvar+dm['dm'].nvar

#     def create_dm(self, par):

#         if numpy.array_equal(self.state,par):
#             return
#         bias = self.data.y

#     # Split into "good" and "bad" samples
#         good_inds = numpy.abs(bias) <= par[0]
#         bad_inds = numpy.logical_not(good_inds)
#         self.good_inds= good_inds
#         self.bad_inds=bad_inds
#         self.config(good_inds, bad_inds, par)



#         self.state = numpy.array(par)

#         train_coordinates = numpy.zeros((len(self.data.x),  self.nvar))

#         # put the new coordinates into a data
#         ncoord=0
#         for dm in self.dm:
#             dm['dm'].make_map()
#             train_coordinates[dm['train_inds'],ncoord:ncoord+dm['dm'].nvar]= \
#                 dm['dm'].internal_coordinates()
#             train_coordinates[numpy.logical_not(dm['train_inds']),ncoord:ncoord+dm['dm'].nvar]= \
#                 dm['dm'].transform(self.data.x[numpy.logical_not(dm['train_inds'])])
#             ncoord=ncoord+dm['dm'].nvar

#         # renormalize the coordinates to be sane
#         self.mns=[]
# #        self.sigs=[]

#         for index in xrange(ncoord):
#             xso=numpy.sort(train_coordinates[:,index])
#             l= len(xso)
#             xso=xso[l*.2:l*.8]
#             xmn = xso[len(xso)/2]
# #            xsig = xso.std()
#             train_coordinates[:,index]=(train_coordinates[:,index]-xmn)#/xsig
#             self.mns.append(xmn)
# #            self.sigs.append(xsig)
#         self.mns=numpy.array(self.mns)

#         xso = numpy.sort(train_coordinates)
#         l=len(xso)
#         xso=xso[l*.2:l*.8]
#         self.sig = xso.std()
#         train_coordinates = train_coordinates/self.sig

#         #self.sigs=numpy.array(self.sigs)
#         self.dmdata = Data(train_coordinates,self.data.y,self.data.z,xlabel=[str(i) for i in xrange(self.nvar)])

#     def coordinates(self, x, par):

#         # if the current state of the diffusion maps is not equal
#         # to what is requested make them
#         self.create_dm(par)

#         coords = numpy.empty((len(x),0))
#         for dm in self.dm:
#             coords=numpy.append(coords, dm['dm'].transform(x),axis=1)

#         for index in xrange(len(self.mns)):
#             coords[:,index]=(coords[:,index]-self.mns[index])/self.sig
#         return coords


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

class MyEstimator(sklearn.base.BaseEstimator):
    """docstring for MyEstimator"""
    def __init__(self, catastrophy_cut=numpy.float_(0.05), eps_val=numpy.float_(0.0025),
        mask_var=numpy.float_(1),xlabel=None,ylabel=None):
        super(MyEstimator, self).__init__()
        self.params=dict()
        self.params['catastrophy_cut']=catastrophy_cut
        self.params['eps_val']=eps_val
        self.params['mask_var']=mask_var


        self.outlier_cut=0.95
        self.optimize_frac = 0.1
        self.xlabel=xlabel
        self.ylabel=ylabel

    def get_params(self,deep=True):
        return self.params

    def set_params(self, catastrophy_cut=None, eps_val=None,
        mask_var=None):
        # for key in self.dict:
        #     self.dict[key]=params[key]
        self.params['catastrophy_cut']=catastrophy_cut
        self.params['eps_val']=eps_val
        self.params['mask_var']=mask_var
        return self

    def fit(self, x, y):

        self.catastrophy = numpy.abs(y) > self.params['catastrophy_cut']



        if False: #os.path.isfile('estimator.pkl'):
            #print 'get pickle'
            pklfile=open('estimator.pkl','r')
            self.dm=pickle.load(pklfile)
        else:
            # the new coordinate system based on the training data
            data = Data(x,y,numpy.zeros(len(y)),xlabel=self.xlabel,ylabel=self.ylabel)
            self.dm=DiffusionMap(data,self.params['eps_val'])
            self.dm.make_map()  
            pklfile=open('estimator.pkl','w')
            pickle.dump(self.dm,pklfile)
        pklfile.close()
        # self.dm=DiffusionMap(x,self.eps_val)
        # self.dm.make_map()



        train_dist = sklearn.metrics.pairwise_distances(self.dm.data_dm().x,self.dm.data_dm().x)
        catastrophy_distances = train_dist[numpy.outer(self.catastrophy,self.catastrophy)]
        catastrophy_distances = catastrophy_distances[catastrophy_distances !=0]
        catastrophy_distances = numpy.sort(catastrophy_distances)
        numpy.fill_diagonal(train_dist,train_dist.max()) #numpy.finfo('d').max)
        train_min_dist = numpy.min(train_dist,axis=0)
        train_min_dist = numpy.sort(train_min_dist)
        self.max_distance = train_min_dist[x.shape[0]*self.outlier_cut]

        self.mask_scale = catastrophy_distances[x.shape[0]*self.params['mask_var']]


    def predict(self, x):
        return 0
    
    def score(self,x,y):
        x_dm = self.dm.transform(x)

        test_dist = sklearn.metrics.pairwise_distances(self.dm.data_dm().x,x_dm)   
        test_min_dist = numpy.min(test_dist,axis=0)

        closer  = test_min_dist < self.max_distance

        self.weight = numpy.exp(-(test_dist[self.catastrophy,:]/self.mask_scale)**2).sum(axis=0)

        weight_sort=numpy.sort(self.weight[closer])
        w=self.weight[closer] <= weight_sort[self.optimize_frac*len(weight_sort)]
#        w=numpy.logical_and(w,closer)

        ans= 1./y[closer[w]].std()
        return ans

    def plots(self,x):

        figax= train_data.plot(color='b',alpha=0.1,s=10)
        train_data.plot(lambda x: self.catastrophy, color='r',alpha=1,s=20,figax=figax)
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
        self.dm.data_dm().plot(lambda x: self.catastrophy,
            color='b',alpha=0.1,s=20,ndim=6,figax=figax)
        plt.savefig('temp.png')
        figax= self.dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
        self.dm.data_dm().plot(lambda x: self.catastrophy,
            color='b',alpha=0.2,s=20,ndim=6,figax=figax)
        plt.savefig('temp2.png')

        cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
        cval=cm.to_rgba(self.weight)
        figax= x.plot(c=cval,alpha=0.2,s=20,cmap=cm,vmin=0,vmax=cval.max())
        plt.savefig('color_dm.png')

if __name__ == '__main__':

    doplot = True
    parser = ArgumentParser()
    parser.add_argument('test_size', nargs='?',default=0.1)
    parser.add_argument('seed', nargs='?',default=9)
    
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    x0 = numpy.array([0.05,0.0025*2.5,0.05])

    optimization_frac = 0.1
    outlier_cut = 0.95

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)


    estimator = MyEstimator(catastrophy_cut=x0[0],
        eps_val=x0[1],mask_var=x0[2],xlabel=train_data.xlabel,ylabel=train_data.ylabel)
    # estimator.fit(train_data, train_data)
    # estimator.score(test_data, test_data.y)
    # estimator.plots(test_data)
    # optimize

    param_grid = [{'catastrophy_cut': numpy.arange(0.03,.08,0.01), 'eps_val': numpy.arange(0.001,0.01,0.001),
        'mask_var': numpy.arange(0.01,0.15,.01)}]

    filename = 'clf.pkl'
    if False:#os.path.isfile(filename):
        #print 'get pickle'
        pklfile=open(filename,'r')
        clf=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        clf = sklearn.grid_search.GridSearchCV(estimator, param_grid, n_jobs=12, cv=10,refit=True)
        clf.fit(train_data.x,train_data.y)
        pklfile=open(filename,'w')
        pickle.dump(clf,pklfile)
    pklfile.close()
    print clf.get_params()
    print clf.score(test_data.x,test_data.y)
    fwef

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

def old():




    dm=DiffusionMap(train_data,x0[1])



    if os.path.isfile('dmsys.pkl'):
        #print 'get pickle'
        pklfile=open('dmsys.pkl','r')
        dm,test_data_dm=pickle.load(pklfile)
    else:
        # the new coordinate system based on the training data
        dm=DiffusionMap(train_data,x0[1])
        dm.make_map()
        test_data_dm=Data(dm.transform(test_data.x),test_data.y,test_data.z,
            xlabel=[str(i) for i in xrange(dm.neigen)])
        pklfile=open('dmsys.pkl','w')
        pickle.dump([dm,test_data_dm],pklfile)
    pklfile.close()


    ## plots that show dm x in color space
    if doplot:
        for i in xrange(6):
            crap = numpy.sort(dm.data_dm().x[:,i])
            crap= crap[len(crap)*.1:len(crap)*.9]
            sig = crap.std()
            cm=matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=crap[len(crap)/2]-5*sig,
                    vmax=crap[len(crap)/2]+5*sig),cmap='Spectral')
            cval=cm.to_rgba(dm.data_dm().x[:,i])
            figax= train_data.plot(c=cval,alpha=0.3,s=20,cmap=cm)
            figax[0].suptitle(str(i))
            plt.savefig('splits.'+str(i)+'.png')

        figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,ndim=6)
        dm.data_dm().plot(lambda x: outliers,
            color='b',alpha=0.5,s=20,ndim=6,figax=figax)
        plt.savefig('temp.png')
        figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
        dm.data_dm().plot(lambda x: outliers,
            color='b',alpha=0.5,s=20,ndim=6,figax=figax)
        plt.savefig('temp2.png')


        # plt.clf()
        # figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
        # test_data.plot(lambda x: w, label='0.2',color='b',alpha=1,s=10,figax=figax)
        # plt.show()

    train_dist = sklearn.metrics.pairwise_distances(dm.data_dm().x,dm.data_dm().x)   
    outlier_distances = train_dist[numpy.outer(outliers,outliers)]
    outlier_distances = outlier_distances[outlier_distances !=0]
    outlier_distances = numpy.sort(outlier_distances)
    norm = outlier_distances[len(outlier_distances)*.05]

    numpy.fill_diagonal(train_dist,train_dist.max()) #numpy.finfo('d').max)
    train_min_dist = numpy.min(train_dist,axis=0)
    closer  = train_min_dist < numpy.sort(train_min_dist)[train_data.ndata()*.95]
    weight = numpy.exp(-(train_dist[outliers,:]/norm)**2).sum(axis=0)
    if doplot:    
        cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
        cval=cm.to_rgba(weight)
        figax= train_data.plot(c=cval,alpha=0.1,s=20,cmap=cm)

    weight_sort=numpy.sort(weight)

    mn=[]
    sd=[]
    fracs=numpy.arange(0.01,0.5,0.05)
    for frac in fracs:
        w=weight < weight_sort[frac*train_data.ndata()]
        w=numpy.logical_and(w,closer)
        mn.append(train_data.y[w].mean())
        sd.append(train_data.y[w].std())
    plt.scatter(fracs,mn,c='b',label='mean')
    plt.scatter(fracs,sd,c='r',label='std')
    plt.show()

    wfe
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

