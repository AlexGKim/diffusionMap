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
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def scatter(X, bias, marker='+', label=None):
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
                marker=marker, label=label)

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
    plt.xlabel('diffusion coordinate 1')
    plt.ylabel('diffusion coordinate 2')
    plt.title('RedMaGiC Diffusion Map: Absolute Magnitudes Only')
    scatter(coords, bias, marker='.')
    plt.colorbar(label='bias')
    if nystrom and nystrombias:
        scatter(nystrom, nystrombias)
    if fname is not None:
        plt.savefig(fname ,format='pdf')
    

# Load training data
f = pyfits.open('../data/stripe82_run_redmagic-1.0-08.fits')
data = f[1].data

# Get rid of entries without spectroscopic redshifts.
inds = data['ZSPEC'] >= 0
sdata = data[inds]

# Compute bias.
bias = sdata['ZRED2'] - sdata['ZSPEC']

# Get features for *entire* sample.
# Absolute magnitudes are the only features I understand, so I use them.
features = sdata['MABS']

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Set a threshold for "good" and "bad" photoz bias
thresh = 0.01

# Split into "good" and "bad" samples
good_inds = abs(bias) <= thresh
bad_inds = np.logical_not(good_inds)
td_good = features[good_inds]
td_bad = features[bad_inds]

# Split into training and testing sets
X_good_train, X_good_test, y_good_train, y_good_test = \
                                    train_test_split(td_good,
                                                     bias[good_inds],
                                                     test_size=0.1)
X_bad_train, X_bad_test, y_bad_train, y_bad_test = \
                                    train_test_split(td_bad,
                                                     bias[bad_inds],
                                                     test_size=0.1)

# Train diffusion maps
dmap_good = diffuse.diffuse(X_good_train)
dmap_bad = diffuse.diffuse(X_bad_train)

# Plot results for good and bad sets.
plot_dmap(dmap_good, 'good_init.pdf', y_good_train)
plot_dmap(dmap_bad, 'bad_init.pdf', y_bad_train)

# Nystrom.

goodtest_baddmap = diffuse.nystrom(dmap_bad, X_good_test)
badtest_baddmap = diffuse.nystrom(dmap_bad, X_bad_test)

goodtest_gooddmap = diffuse.nystrom(dmap_good, X_good_test)
badtest_gooddmap = diffuse.nystrom(dmap_good, X_bad_test)

plot_dmap(dmap_good, None, y_good_train)
scatter(goodtest_gooddmap, y_good_test, marker='+', label='low bias')
scatter(badtest_gooddmap, y_bad_test, marker='^', label='high bias')
plt.savefig('good_test.pdf', format='pdf')

plot_dmap(dmap_bad, None, y_bad_train)
scatter(goodtest_baddmap, y_good_test, marker='+', label='low bias')
scatter(badtest_baddmap, y_bad_test, marker='^', label='high bias')
plt.savefig('bad_test.pdf', format='pdf')
