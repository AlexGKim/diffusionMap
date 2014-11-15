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
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser

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
    coords=dmap['X']
    plt.xlabel('diffusion coordinate 1')
    plt.ylabel('diffusion coordinate 2')
    plt.title('RedMaGiC Diffusion Map: Absolute Magnitudes Only')
    scatter(coords, bias, marker='.')
    plt.colorbar(label='bias')
    if nystrom and nystrombias:
        scatter(nystrom, nystrombias)
    if fname is not None:
        plt.savefig(fname ,format='pdf')
    

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-eps_val', help='steps taken in diffusion process',default='default')
    parser.add_argument('--save', action='store_true', help='Pickle trained diffusion maps.')
    parser.add_argument('--load', action='store_true', help='Load trained diffusion maps from pickles.')
    
    ins = parser.parse_args()
    pdict=vars(ins)

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
    features = np.hstack((features, sdata['MABS'][:, 2].reshape(-1, 1))) # i magnitude

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
                                                         test_size=0.1,
                                                         random_state=9)
    X_bad_train, X_bad_test, y_bad_train, y_bad_test = \
                                        train_test_split(td_bad,
                                                         bias[bad_inds],
                                                         test_size=0.1,
                                                         random_state=10)

    # Train diffusion maps
    if not ins.load:
        kwargs=dict()
        kwargs['delta']=1e-8
        if pdict['eps_val'] != 'default':
           kwargs['eps_val'] = float(pdict['eps_val'])
        kwargs['t']=1
        dmap_good = diffuse.diffuse(X_good_train, **kwargs)
        dmap_bad = diffuse.diffuse(X_bad_train, **kwargs)

    else:
        dmap_good = pickle.load(open('dmap_good.obj','rb'))
        dmap_bad = pickle.load(open('dmap_bad.obj','rb'))

    # Save if asked
    if ins.save:
        pickle.dump(dmap_good, open('dmap_good.obj','wb'))
        pickle.dump(dmap_bad, open('dmap_bad.obj','wb'))

    

    # Plot results for good and bad sets.
    plot_dmap(dmap_good, 'good_init.'+pdict['eps_val']+'.pdf', y_good_train)
    plot_dmap(dmap_bad, 'bad_init.'+pdict['eps_val']+'.pdf', y_bad_train)

    # Nystrom.

    goodtest_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_good_test))
    badtest_baddmap = np.array(diffuse.nystrom(dmap_bad, X_bad_train, X_bad_test))

    goodtest_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_good_test))
    badtest_gooddmap = np.array(diffuse.nystrom(dmap_good, X_good_train, X_bad_test))
#    plot_dmap(dmap_good, None, y_good_train)
    plt.clf()
#    scatter(goodtest_gooddmap, y_good_test, marker='+', label='low bias')
#    scatter(badtest_gooddmap, y_bad_test, marker='^', label='high bias')

#    plt.show()
#    plt.clf()
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X=goodtest_gooddmap
    ax.scatter(X.T[0],X.T[1],X.T[2],marker='D',c='b')
    X=badtest_gooddmap
    ax.scatter(X.T[0],X.T[1],X.T[2],marker='^',c='r')
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_zlabel('X[2]')
#    ar = numpy.append(goodtest_gooddmap.T[0],badtest_gooddmap.T[0])
#    mn = numpy.mean(ar)
#    sd = numpy.std(ar)
#    ax.set_xlim((mn-5*sd,mn+5*sd))
#    ar = numpy.append(goodtest_gooddmap.T[1],badtest_gooddmap.T[1])
#    mn = numpy.mean(ar)
#    sd = numpy.std(ar)
#    ax.set_ylim((mn-5*sd,mn+5*sd))
#    ar = numpy.append(goodtest_gooddmap.T[2],badtest_gooddmap.T[2])
#    mn = numpy.mean(ar)
#    sd = numpy.std(ar)
#    ax.set_zlim((mn-5*sd,mn+5*sd))
    plt.savefig('good_test.'+pdict['eps_val']+'.pdf', format='pdf')

    plt.clf()
    fig=plt.figure()
#    plot_dmap(dmap_bad, None, y_bad_train)
#    scatter(goodtest_baddmap, y_good_test, marker='+', label='low bias')
#    scatter(badtest_baddmap, y_bad_test, marker='^', label='high bias')
#    plt.show()
#    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    X=goodtest_baddmap
    ax.scatter(X.T[0],X.T[1],X.T[2],marker='D',c='b')
    X=badtest_baddmap
    ax.scatter(X.T[0],X.T[1],X.T[2],marker='^',c='r')
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_zlabel('X[2]')

    ar = numpy.sort(numpy.append(goodtest_baddmap.T[0],badtest_baddmap.T[0]))[20:-20]
    mn = numpy.mean(ar)
    sd = numpy.std(ar)
    print (mn-5*sd,mn+5*sd)
    ax.set_xlim((mn-5*sd,mn+5*sd))
    ar = numpy.sort(numpy.append(goodtest_baddmap.T[1],badtest_baddmap.T[1]))[20:-20]
    mn = numpy.mean(ar)
    sd = numpy.std(ar)
    print (mn-5*sd,mn+5*sd)
    ax.set_ylim((mn-5*sd,mn+5*sd))
    ar = numpy.sort(numpy.append(goodtest_baddmap.T[2],badtest_baddmap.T[2]))[20:-20]
    mn = numpy.mean(ar)
    sd = numpy.std(ar)
    print (mn-5*sd,mn+5*sd)
    ax.set_zlim((mn-5*sd,mn+5*sd))
    plt.savefig('bad_test.'+pdict['eps_val']+'.pdf', format='pdf')
