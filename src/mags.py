#!/usr/bin/env python

'''Create a corner plot of the Stripe82 magnitude dataset.'''

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits
import triangle
import numpy as np

data = pyfits.open('../data/stripe82_run_redmagic-1.0-08.fits')[1].data

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

# Plot. 

fig = triangle.corner(features, labels=['g-r', 'r-i', 'i-z', 'i'], extents=[.999] * 4)

fig.savefig('color-mag.pdf', format='pdf')
