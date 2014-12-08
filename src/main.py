#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'


#import pickle
#import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy
from argparse import ArgumentParser
import diffuse


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


class Plots:
    @staticmethod
    def diffusionMaps(dmsystem):
        dmlabels=[]
        for dm in dmsystem.dm:
            dmlabels.append(dm.label)
        dmlabels=numpy.array(dmlabels)
        zlabels  = numpy.unique(dmlabels[:,0])
        pp = PdfPages('temp.pdf')
        #not right, want good/bad for each of the permutations
        for zlabel in zlabels:
            plt.clf()
            fig=plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            w= numpy.where(numpy.logical_and(dmlabels[:,0] == zlabel, dmlabels[:,1]=='good'))[0]
            dmsystem.dm[w].plot(ax,marker='D',c='b')
            w= numpy.where(numpy.logical_and(dmlabels[:,0] == zlabel, dmlabels[:,1]=='bad'))[0]
            dmsystem.dm[w].plot(ax,marker='^',c='r')
            pp.savefig()
        pp.close()
        print shit

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
    return Data(X_train, y_train, z_train), Data(X_test,  y_test, z_test,xlabel=['g-r','r-i','i-z','i'], ylabel='bias',zlabel='photo-z')


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
       self.label=label

    def make_map(self):

        """ Method that calculates the diffusion map.  The result is
        stored internally.
        """

        kwargs=dict()
        kwargs['eps_val'] = self.par.item()
        kwargs['t']=1
        kwargs['delta']=1e-8
#        print self.data.x.shape
        self.dmap = diffuse.diffuse(self.data.x, **kwargs)

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

        return diffuse.nystrom(self.dmap, self.data.x, x)

    def plot(self,ax, oplot=False, **kwargs):

        X=self.dmap['X']
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

    def __init__(self, data):
        self.nbins =4
        self.data = data 
        self._redshift_bins()
        self.state = None
        
    def _redshift_bins(self):
        """ Calculates the photometric redshift bin ranges and
        determine which subsets of Data go into which bin, the results
        of which are set in self.wzbins
        """
        zp = numpy.log(1+self.data.z)
        lzmax = numpy.log(1+zmax)
        lzmin = numpy.log(1+zmin)
        
        delta = (lzmax-lzmin+1e-16)/self.nbins

        self.wzbins=[]

        for i in xrange(self.nbins):
            dum = zp >= lzmin+i*delta
            self.wzbins.append(numpy.logical_and(dum,zp< lzmin+(i+1)*delta))

    def create_dm(self, par):
        
        bias = self.data.y

    # Split into "good" and "bad" samples
        good_inds = abs(bias) <= par[0]
        bad_inds = numpy.logical_not(good_inds)

        self.dm=[]
        
        for wzbin,zlab in zip(self.wzbins,xrange(len(self.wzbins))):
            for wbbin,blab in zip([good_inds,bad_inds],['good','bad']):
                wboth = numpy.logical_and(wzbin,wbbin)
                if len(wboth) == 0:
                    raise Exception('Nothing in training set')
                #print par
                newmap = DiffusionMap(self.data[wboth],par[1],numpy.array([zlab,blab]))
                self.dm.append(newmap)

    def train(self):
        for dm in self.dm:
            dm.make_map()


    def coordinates(self, x, par):

        # if the current state of the diffusion maps is not equal
        # to what is requested make them
        if not numpy.array_equal(par,self.state):
            self.create_dm(par)
            self.train()

        coords = numpy.empty((len(x),0))
        for dm in self.dm:
            coords=numpy.append(coords, dm.transform(x),axis=1)

        return coords

class WeightedBias:

    import sklearn

    def __init__(self, dmsys, random_state=10):
        self.dmsys = dmsys
        self.random_state = random_state
        import sklearn.ensemble
        self.classifier = sklearn.ensemble.RandomForestClassifier(random_state=self.random_state)

    def biases(self):
        return self.dmsys.data.y

    def rule(self,par):
        dmcoords = self.dmsys.coordinates(self.dmsys.data.x, par)
        self.classifier.fit(dmcoords, self.dmsys.data.lessthan(par[0]))
        ans=self.classifier.predict(dmcoords)
        print type(ans), ans.shape
        print ans
        return ans

    def value(self,par):
        w = self.rule(par)
        if len(w) == 0:
            raise Exception("No passing objects")
        else:
            return numpy.sum(self.biases()[w])**2/len(w)
 
def train(wb):

    # the things to optimize are eps_val and threshold for good and bad
    fun=wb.value
    x0 = numpy.array([0.01,0.001])
    import scipy.optimize
    ans = scipy.optimize.minimize(fun,x0)
    print ans
    return
    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('test_size', nargs='?',default=0.1)
    parser.add_argument('seed', nargs='?',default=9)
    
    ins = parser.parse_args()
    pdict=vars(ins)

    rs = numpy.random.RandomState(pdict['seed'])

    # data
    train_data, test_data = manage_data(pdict['test_size'],rs)

    # the new coordinate system based on the training data
    dmsys= DMSystem(train_data)
    x0 = numpy.array([0.01,0.001])
    dmsys.create_dm(x0)
    dmsys.train()

    Plots.diffusionMaps(dmsys)

    shit
    # the calculation of the weighted bias
    wb = WeightedBias(dmsys, rs)

    # optimization
    train(wb)

    shit




