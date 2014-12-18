#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'


import pickle
#import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy
from argparse import ArgumentParser
import diffuse
from scipy.stats import norm
import sklearn
import sklearn.ensemble
from matplotlib.backends.backend_pdf import PdfPages


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

class Plots:
    @staticmethod
    def diffusionMaps(dmsystem,x0,nsig=20):
        dmz = [dm['zbin'] for dm in dmsystem.dm]
        dmb = [dm['bias'] for dm in dmsystem.dm]
        zs = numpy.unique(dmz)

#        pp = PdfPages('dm.pdf')
        figax=[]
        for z in zs:
            wg=numpy.where(numpy.logical_and(dmz == z,[b==True for b in dmb]))[0]
            wb=numpy.where(numpy.logical_and(dmz == z,[b==False for b in dmb]))[0]
            for b in [True, False]:
                #plt.clf()
                #fig=plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                if b:
                    figax.append(Plots.oneDiffusionMap(dmsystem,wg,x0,nsig=nsig))
#                    dmsystem.dm[wg]['dm'].plot(ax,marker='D',c='b')
#                    dmsystem.dm[wg]['dm'].plot_external(ax,dmsystem.dm[wb]['dm'].data.x,marker='^',c='r',oplot=True)
                else:
                    figax.append(Plots.oneDiffusionMap(dmsystem, wb, x0,nsig=nsig))
#                    dmsystem.dm[wb]['dm'].plot(ax,marker='^',c='r')
#                    dmsystem.dm[wb]['dm'].plot_external(ax,dmsystem.dm[wg]['dm'].data.x,marker='D',c='b',oplot=True)
#                plt.title(str(z)+" "+str(b))
 #               pp.savefig()
#        pp.close()
        return figax

    @staticmethod
    def oneDiffusionMap(dmsys,ind, x0,nsig=20):
        alpha=0.025
        s=5
        ndim=4
        dm=dmsys.dm[ind]['dm']
        isgood = dmsys.dm[ind]['bias']
        figax = plt.subplots(nrows=ndim-1,ncols=ndim-1,figsize=(8,6))
        
        if isgood:
            badx=dmsys.data.x[numpy.abs(dmsys.data.y) > x0[0]]
            bx = dm.transform(badx)
            gx = dm.dmap['X']
            by= dmsys.data.y[numpy.abs(dmsys.data.y) > x0[0]]
        else:
            goodx=dmsys.data.x[numpy.abs(dmsys.data.y) <= x0[0]]
            gx = dm.transform(goodx)
            bx = dm.dmap['X']
            by = dm.data.y
            badx= dm.data.x[numpy.abs(dm.data.y) > x0[0]]
                #hasanalog=DMSystem.hasTrainingAnalog(x)
        figax[0].suptitle(str(isgood))
        Plots.x(gx[:,xrange(ndim)],figax=figax,label='low bias',color='g',alpha=alpha,s=s)
        w=split(badx)
        dum = bx[:,xrange(ndim)]
        w_=numpy.logical_and(w,  numpy.abs(by) <= x0[0])
        Plots.x(dum[w,:],figax=figax,label='high bias: Pop 1',color='b',s=s,alpha=alpha)
        w_=numpy.logical_and(w,  numpy.abs(by) > x0[0])
        Plots.x(dum[w,:],figax=figax,label='high bias: Pop 2',color='r',s=s,alpha=alpha)

        for ic in xrange(ndim-1):
            for ir in xrange(ic,ndim-1):
                l = len(dm.dmap['X'][:,ic])
                #l = w.sum()
                xl = numpy.sort(dm.dmap['X'][:,ic])
                xmn = xl[int(l*.5)]
                xsd = xl[int(l*.05):int(l*.95)].std()
                xl = numpy.sort(dm.dmap['X'][:,ir+1])
                ymn = xl[int(l*.5)]
                ysd = xl[int(l*.05):int(l*.95)].std()

                figax[1][ir,ic].set_xlim(xmn-nsig*xsd,xmn+nsig*xsd)
                figax[1][ir,ic].set_ylim(ymn-nsig*ysd,ymn+nsig*ysd)

        return figax

    @staticmethod
    def data(data,par,**kwargs):
        alpha=0.025
        s=5
        marker='.'
        ndim=data.x.shape[1]

        figax = plt.subplots(nrows=ndim-1,ncols=ndim-1,figsize=(8,6)
)

        Plots.x(data.x[numpy.abs(data.y) <= par[0],:],figax=figax,alpha=alpha,s=s,marker=marker, color='g',label='low bias',xlabel=data.xlabel)
        Plots.x(data.x[numpy.abs(data.y) > par[0],:],figax=figax,alpha=alpha,s=s,marker=marker, color='b',label='high bias: Pop 1',xlabel=data.xlabel)
        w = numpy.logical_and(split(data.x), numpy.abs(data.y) > par[0])
        Plots.x(data.x[w,:],figax=figax,alpha=alpha,s=s,marker=marker, label='hight bias: Pop 2',color='r',xlabel=data.xlabel)


    @staticmethod
    def x(x,good='default',xlabel=None, figax='default',**kwargs):

        ndim=x.shape[1]-1
        if figax == 'default':
            fig, axes = plt.subplots(nrows=ndim,ncols=ndim)
        else:
            fig,axes = figax


        for i in xrange(axes.shape[0]):
            for j in xrange(axes.shape[1]):
                axes[i,j].set_visible(False)

        for ic in xrange(ndim):
            for ir in xrange(ic,ndim):
                axes[ir,ic].set_visible(True)
                axes[ir,ic].scatter(x[:,ic],x[:,ir+1],**kwargs)
                

                                            
                axes[ir,ic].legend(prop={'size':6})
                if xlabel is not None:
                    if ic==0:
                        axes[ir,ic].set_ylabel(xlabel[ir+1])
                    if ir==ndim-1:
                        axes[ir,ic].set_xlabel(xlabel[ic])
                if ic != 0:
                    axes[ir,ic].get_yaxis().set_visible(False)
                if ir !=ndim-1:
                    axes[ir,ic].get_xaxis().set_visible(False)


        


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

    def plot(self):
        import triangle
        figure  = triangle.corner(self.x)
        return figure

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
    inds = data['ZSPEC'] >= 0
    sdata = data[inds]

    # Get rid of entries outside of photo-z range
    inds = numpy.logical_and(sdata['ZRED2'] >= zmin, sdata['ZRED2'] < zmax)
    sdata = sdata[inds]

    # Compute bias.
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

        if config == 'default':
            self.config = self._onebad
        else:
            self.config = self._redshift_bins

    def _onebad(self, good_inds, bad_inds, par):
        self.dm=[]
        key = dict()
        newmap=DiffusionMap(self.data[bad_inds],par[1],key)
        key['dm']=newmap
        key['bias']=False
        key['zbin']=0
        self.dm.append(key)

        key = dict()
        newmap=DiffusionMap(self.data[good_inds],par[1],key)
        key['dm']=newmap
        key['bias']=True
        key['zbin']=0
        self.dm.append(key)

        
    def _redshift_bins(self, good_inds, bad_inds, par):
        """ Calculates the photometric redshift bin ranges and
        determine which subsets of Data go into which bin, the results
        of which are set in self.wzbins
        """
        self.nbins =4

        zp = numpy.log(1+self.data.z)
        lzmax = numpy.log(1+zmax)
        lzmin = numpy.log(1+zmin)
        
        delta = (lzmax-lzmin+1e-16)/self.nbins

        self.wzbins=[]

        for i in xrange(self.nbins):
            dum = zp >= lzmin+i*delta
            self.wzbins.append(numpy.logical_and(dum,zp< lzmin+(i+1)*delta))

        self.dm=[]
        
        for wzbin,zlab in zip(self.wzbins,xrange(len(self.wzbins))):
           for wbbin,blab in zip([good_inds,bad_inds],[True,False]):
                wboth = numpy.logical_and(wzbin,wbbin)
                if len(wboth) == 0:
                    raise Exception('Nothing in training set')
                key=dict()
                key['zbin']=zlab
                key['bias']=blab
                newmap = DiffusionMap(self.data[wboth],par[1],key)
                key['dm']=newmap
                self.dm.append(key)

    def create_dm(self, par):

        if numpy.array_equal(self.state,par):
            return
        bias = self.data.y

    # Split into "good" and "bad" samples
        good_inds = numpy.abs(bias) <= par[0]
        bad_inds = numpy.logical_not(good_inds)
        self.config(good_inds, bad_inds, par)


        for dm in self.dm:
            dm['dm'].make_map()
        self.state = numpy.array(par)

    def coordinates(self, x, par):

        # if the current state of the diffusion maps is not equal
        # to what is requested make them
        self.create_dm(par)

        coords = numpy.empty((len(x),0))
        for dm in self.dm:
            coords=numpy.append(coords, dm['dm'].transform(x),axis=1)

        return coords

class WeightedBias:

    def __init__(self, dmsys, random_state=10):
        self.nuse=4
        self.dmsys = dmsys
        self.random_state = random_state
        self.classifier = sklearn.ensemble.RandomForestClassifier(random_state=self.random_state)

    def train(self,par):
        dmcoords = self.dmsys.coordinates(self.dmsys.data.x, par)
        y = numpy.array(self.dmsys.data.y)

        ok = DMSystem.hasTrainingAnalog(dmcoords)

        self.classifier.fit(dmcoords[ok,0:self.nuse], numpy.abs(y[ok]) <= par[0])
        self.dmcoords=dmcoords
        self.hasAnalog=ok
#        return dmcoords, y

    def weighted_mean(self, dmcoords, y, par):
        ok = DMSystem.hasTrainingAnalog(dmcoords)
        goodin=numpy.where(self.classifier.classes_)[0][0]

        proba= self.classifier.predict_proba(dmcoords[ok,0:self.nuse])[:,goodin]
        ans=numpy.empty(len(y),dtype='bool')
        ans.fill(True)
        ans[ok] = proba > 0.9
        if numpy.sum(ans) == 0:
            raise Exception("No passing objects")
        else:
            res= numpy.mean(y[ans])
            print res, y[ans].std(), numpy.sum(ans),
            res = res**2/numpy.sum(ans)
            print res
            return res

    def value_internal(self, par):
        print par
        self.train(par)
        return self.weighted_mean(self.dmcoords[self.hasAnalog],self.dmsys.y[self.hasAnalog],par)

    def value_external(self, x_, y_, par):
        self.train(par)
        dmcoords = self.dmsys.coordinates(x_, par)

        return self.weighted_mean(dmcoords,y_,par)

 
def train(wb):

    # the things to optimize are eps_val and threshold for good and bad
    fun=wb.value_internal
    #x0 = numpy.array([0.015,0.001])
    import scipy.optimize
    ans = scipy.optimize.brute(fun,((0.01,0.04),(5e-4,3e-2)),finish=scipy.optimize.fmin)
#    print ans[0], type(ans[0])
#    ans2=  scipy.optimize.minimize(fun,ans[0],bounds=[(0.01,0.1),(5e-4,3e-2)])
    print ans
    return

def colorClassify(train_data,test_data,par):
    ans = scipy.optimize.brute(fun,((0.01,0.04),(5e-4,3e-2)),finish=None)
    n_estimators=20, max_features=[2,8]
    classifier = sklearn.ensemble.RandomForestClassifier(random_state=10)
    classifier.fit(train_data.x, numpy.abs(train_data.y) <= par[0])
    goodin=numpy.where(classifier.classes_)[0][0]
    proba=classifier.predict_proba(test_data.x)[:,goodin]
    ans = proba > 0.9
    if numpy.sum(ans) == 0:
        raise Exception("No passing objects")
    else:
        res= numpy.mean(test_data.y[ans])
        print res, test_data.y[ans].std(), numpy.sum(ans),
        res = res**2/numpy.sum(ans)
        print res
        return res

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('test_size', nargs='?',default=0.1)
    parser.add_argument('seed', nargs='?',default=9)
    
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    x0 = numpy.array([0.02,0.004])

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)
    print test_data.y.mean(), test_data.y.std(), len(test_data.y)

    colorClassify(train_data, test_data, x0)


    # the new coordinate system based on the training data
    dmsys= DMSystem(train_data)
#    plt.clf()
#    Plots.data(dmsys.data,x0)
#    plt.savefig('x.pdf')
#    we
    for x01 in [0.0025]:
        x0[1]=x01
        dmsys.create_dm(x0)
        plt.clf()
 #   figax=Plots.oneDiffusionMap(dmsys,0,x0)
#    plt.savefig('temp.pdf')
        pp = PdfPages('dms.blowup.'+str(x01)+'.pdf')
        figax=Plots.diffusionMaps(dmsys,x0,nsig=5)
        pp.savefig(figax[0][0])
        pp.savefig(figax[1][0])
        pp.close()
    wfe
    # the calculation of the weighted bias
    wb = WeightedBias(dmsys, rs)
    # optimization
#    t=train(wb)
#    Plots.diffusionMaps(dmsys)
    wb.value_external(test_data.x, test_data.y, x0)
#    pickle.dump([t,dmsys], open("trained_dmsystem.pkl","wb"))

    shit
