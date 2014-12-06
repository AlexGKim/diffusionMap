#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'

import pyfits
import pickle
import diffuse
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
import numpy
#import matplotlib.pyplot as plt
from argparse import ArgumentParser
#from mpl_toolkits.mplot3d import Axes3D

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

    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z

    def __getitem__(self , index):
        return Data(self.x[index],self.y[index],self.z[index])

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
    return Data(X_train, y_train, z_train), Data(X_test,  y_test, z_test)


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

    def __init__(self, data, par):
       self.data = data
       self.par = par  #for the moment eps_val

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
        
        for wzbin in self.wzbins:
            for wbbin in [good_inds,bad_inds]:
                wboth = numpy.logical_and(wzbin,wbbin)
                if len(wboth) == 0:
                    raise Exception('Nothing in training set')
                newmap = DiffusionMap(self.data[wboth],par[1])
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
            print  dm.transform(x).shape,            
            coords=numpy.append(coords, dm.transform(x),axis=1)
            print coords.shape

        return coords

class WeightedBias:

    def __init__(self, data):
        self.data = data

    def biases(self):
        return self.data.y

    def rule(self,par):
        return numpy.arange(numpy.round(par[0]*len(self.data.y)),dtype='int')

    def value(self,par):
        w = self.rule(par)
        if len(w) == 0:
            raise Exception("No passing objects")
        else:
            return numpy.sum(self.biases()[w])**2/len(w)
 
def train(wb):

    # the things to optimize are eps_val and threshold for good and bad
    fun=wb.value
    x0 = numpy.zeros(1)+.2
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

    # data
    train_data, test_data = manage_data(pdict['test_size'],pdict['seed'])


    par=numpy.array((0.01,1e-2))

    dmcoord= DMSystem(train_data)
    dmcoord.coordinates(train_data.x,par)
    stop
    wb = WeightedBias(train_data)
    train(wb)

    shit

    # Nystrom.
    goodtest_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_good_test))
    goodtrain_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_good_train))
    badtrain_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_bad_train))
    badtest_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_bad_test))

    goodtest_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_good_test))
    badtest_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_bad_test))
    badtrain_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_bad_train))
    goodtrain_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_good_train))

    scatter_3d(goodtrain_baddmap,badtrain_baddmap)
    plt.savefig('good_train.'+pdict['eps_val']+'.pdf', format='pdf')

    scatter_3d(goodtrain_gooddmap,badtrain_gooddmap)
    plt.savefig('bad_train.'+pdict['eps_val']+'.pdf', format='pdf')

#    plot_dmap(dmap_good, None, y_good_train)
    scatter_3d(goodtest_gooddmap,badtest_gooddmap)
    plt.savefig('good_test.'+pdict['eps_val']+'.pdf', format='pdf')
#    scatter(goodtest_gooddmap, y_good_test, marker='+', label='low bias')
#    scatter(badtest_gooddmap, y_bad_test, marker='^', label='high bias')

    scatter_3d(goodtest_baddmap,badtest_baddmap)
    plt.savefig('bad_test.'+pdict['eps_val']+'.pdf', format='pdf')

# use badtrain_*map as the data for "bad", and goodtrain_*map as the data for "good" to establish the basis for the random forest.
# performance will be based on data of *test_*map



