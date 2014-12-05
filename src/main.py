#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

import pyfits
import pickle
import diffuse
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
import numpy
#import matplotlib.pyplot as plt
from argparse import ArgumentParser
#from mpl_toolkits.mplot3d import Axes3D

zmin = 0.1
zmax = 0.8

class Data:

    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z

    def __getitem__(self , index):
#        print self.x.shape, len(index),self.x[index].shape
        return Data(self.x[index],self.y[index],self.z[index])

# reads in data, get subset of data with redshifts, and splits into
# test and training sets

def manage_data(test_size, random_state):
    # Load training data
    f = pyfits.open('../data/stripe82_run_redmagic-1.0-08.fits')
    data = f[1].data

    # Get rid of entries without spectroscopic redshifts.
    inds = data['ZSPEC'] >= 0
    sdata = data[inds]

    # Compute bias.
    bias = sdata['ZRED2'] - sdata['ZSPEC']

    # Get features for *entire* sample.
    # Use G-R, R-I, I-Z colors and I absolute magnitude as features.
    features = sdata['MABS'][:, :-1] - sdata['MABS'][:, 1:] # colors
    features = numpy.hstack((features, sdata['MABS'][:, 2].reshape(-1, 1))) # i magnitude

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test, z_train, z_test = \
                                        train_test_split(features_scaled,
                                                         bias,
                                                         sdata['ZSPEC'],
                                                         test_size=test_size,
                                                         random_state=random_state)
    return Data(X_train, y_train, z_train), Data(X_test,  y_test, z_test)


# Diffusion map
class DiffusionMap:

    def __init__(self, data, par):
       self.data = data
       self.par = par  #for the moment eps_val

    def make_map(self):
        kwargs=dict()
        kwargs['eps_val'] = self.par)
        kwargs['t']=1
        self.dmap = diffuse.diffuse(self.data.x, **kwargs)

    def transform(self, x):
        return diffuse.nystrom(self.dmap, self.data.x, x)

# New coordinate system is based on a set of diffusion maps
class DMCoordinates:

    def __init__(self, data):
        self.nbins = 4
        self.data = data
        self.redshift_bins()
        
    def redshift_bins(self):
        zp = numpy.log(1+self.data.z)
        lzmax = numpy.log(1+DMCoordinates.zmax)
        lzmin = numpy.log(1+DMCoordinates.zmin)
        
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

    def coordinates(self, x, par):
        self.create_dm(par)
        for dm  in self.dm:
            print dm.data.z.shape
        

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
    parser.add_argument('seed', nargs='?',default=4)
    
    ins = parser.parse_args()
    pdict=vars(ins)

    # data
    train_data, test_data = manage_data(pdict['test_size'],pdict['seed'])

    # parameters
    #par[0] = threshold in bias that defines good and bad
    #par[1] = eps_val
    par=numpy.array((0.01,1e-2))

    dmcoord= DMCoordinates(train_data)
    dmcoord.coordinates(None,par)
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



