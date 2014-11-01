import pyfits
import pickle
import diffuse

f = pyfits.open('../data/stripe82_run_redmagic-1.0-08.fits')
td = pickle.load(open('../data/training_data.pkl','rb'))
data = f[1].data
zdiff = abs(data['ZSPEC'] - data['ZRED2'])
inds = data['ZSPEC'] >= 0
zdiff_tr = zdiff[inds]
dmap = diffuse.diffuse(td)


import matplotlib
matplotlib.use('Agg')

import numpy as np
coords = np.array(dmap.rx('X')[0])

import matplotlib.pyplot as plt
plt.scatter(coords.T[0], coords.T[1], c=zdiff[inds], norm=plt.matplotlib.colors.Normalize(vmin=0, vmax=.1), cmap=plt.matplotlib.cm.jet)
plt.colorbar(label='bias')
plt.xlabel('diffusion coordinate 1')
plt.ylabel('diffusion coordinate 2')
plt.title('RedMaGiC Diffusion Map: 19 Input Features')
plt.savefig('19test.pdf',format='pdf')


