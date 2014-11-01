#!/usr/bin/env python

'''Test the `diffuse` package using the toy 'target' dataset.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
__contributors__ = ['Danny Goldstein <dgold@berkeley.edu>']

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import diffuse
import os

def to_colors(class_arr):
  '''Transform an array of class labels into a list of colors.'''
  
  print class_arr
  uq = np.unique(class_arr)
  cdict = dict(zip(uq, mpl.rcParams['axes.color_cycle'][:len(uq)]))
  return [cdict[elem] for elem in class_arr]
  
# Input files.
TASK_NAME = 'Tetra'
FCPS_DIR = '../data/FCPS/01FCPSdata'
CLASS_FILE = os.path.join(FCPS_DIR, '%s.cls' % TASK_NAME)
DATA_FILE = os.path.join(FCPS_DIR, '%s.lrn' % TASK_NAME)



# Load classes, data.  The first column of each file is a meaningless
# 1-based index, so we drop it.
classes = np.squeeze(np.genfromtxt(CLASS_FILE, comments='%')[:, 1:])
data = np.squeeze(np.genfromtxt(DATA_FILE, comments='%')[:, 1:])

# Map classes to colors.
colors = to_colors(classes)

# Compute diffusion map.
kwargs = {}

dmap = diffuse.diffuse(data, **kwargs)

# Extract new representation.
X = np.array(dmap.rx('X')[0])

# Plot. 
'''
fig, ax = plt.subplots()
x, y = X[:, :2].T
ax.scatter(x, y, c=colors)
ax.set_xlim(x[x >= -6e14].min(), x[x >= -6e14].max())

# Save.
fig.savefig('diffmap.pdf', format='pdf')
'''


# Plot 3D.
x, y, z = X[: , :3].T

from mpl_toolkits.mplot3d import Axes3D
with PdfPages('multipage_diffmap.pdf') as pdf:
  for i in range(3):
    for j in range(3):
      figure = plt.figure(figsize=(6,6))
      ax = plt.subplot(111, projection='3d')
      ax.scatter(x, y, z, c=colors)
      ax.view_init(120 * i, 120 * j)
      pdf.savefig(figure)
      plt.close()

