#!/usr/bin/env python

'''Training and testing diffusion maps for the redmagic classification
task.'''

__author__ = 'Alex Kim <agkim@lbl.gov>'
import sys
import os
import os.path
import pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerNpoints
#from matplotlib.backends.backend_pdf import PdfPages
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
from sklearn.metrics import  make_scorer
import diffuse
import copy
from scipy.stats import norm
from scipy.stats.mstats import moment
from redmagic import *

def generic(data):
    plt.scatter(data.z,data.y)
    plt.show()
    wef

if __name__ == '__main__':

    from sklearn.externals import joblib

    rs = numpy.random.RandomState(9)
    train_data, test_data = manage_data(0.9,rs)


#    generic(train_data)

    # filename = os.environ['SCRATCH']+'/diffusionMap/results/clf_mpi_color.pkl'
    # clf_c = joblib.load(filename)

    filename = '/Users/akim/project/diffusionmap/results/clf_mpi.pkl'
    clf = joblib.load(filename)



    #print clf.best_params_
    clf.best_estimator_.plots(train_data)


    wefwef
    clf_c.best_estimator_.plots(test_data)
    #c
    # print clf.best_params_
    # print clf.score(test_data.x,test_data.y)
    #get distances
 #    estimator.fit(train_data.x, train_data.y)
    # estimator.score(test_data, test_data.y)
    # estimator.plots(test_data.x)
    # optimize

    sdfsd




#     class OneOutput(object):
#         """docstring for OneOutput"""
#         def __init__(self, feature_importances_,y,dy2):
#             super(OneOutput, self).__init__()
#             self.feature_importances_ = feature_importances_
#             self.y=y
#             self.dy2=dy2
            
#     class Output(object):
#         """docstring for Output"""
#         def __init__(self,test_min_dist,train_cut, outputs):
#             super(Output, self).__init__()
#             self.test_min_dist=test_min_dist
#             self.train_cut=train_cut
#             self.outputs=outputs
#     filename='output.pkl'

#     if os.path.isfile(filename):
#         #print 'get pickle'
#         pklfile=open(filename,'r')
#         coloroutput,dmoutput=pickle.load(pklfile)
#     else:
#         nrealize=100
#         train_dist = sklearn.metrics.pairwise_distances(train_data.x,train_data.x)
#         numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
#         train_min_dist=numpy.min(train_dist,axis=0)
#         train_sort = numpy.argsort(train_min_dist)
#         train_sort = train_sort[0:prunefrac * len(train_sort)]
#         train_cut =  train_min_dist[train_sort[-1]]
#         test_dist = sklearn.metrics.pairwise_distances(train_data.x[train_sort],test_data.x)
#         test_min_dist = numpy.min(test_dist,axis=0)

#         clf = sklearn.ensemble.forest.RandomForestRegressor(n_estimators=100,random_state=12)
#         color_outs=[]
#         for rs in xrange(nrealize):
#             clf.set_params(random_state=rs)
#             clf.fit(train_data.x[train_sort],train_data.y[train_sort])
#             y,dy2 = clf.predict(test_data.x)
#             color_outs.append(OneOutput(clf.feature_importances_,y,dy2))
#         coloroutput=Output(test_min_dist,train_cut,color_outs)

#         train_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,0:4],dmsys.dmdata.x[:,0:4])**2+
#             sklearn.metrics.pairwise_distances(dmsys.dmdata.x[:,4:],dmsys.dmdata.x[:,4:])**2)
#         numpy.fill_diagonal(train_dist,numpy.finfo('d').max)
#         train_min_dist=numpy.min(train_dist,axis=0)
#         train_sort = numpy.argsort(train_min_dist)
#         train_sort = train_sort[0:prunefrac * len(train_sort)]
#         train_cut =  train_min_dist[train_sort[-1]]
#         test_dist = numpy.sqrt(sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,0:4],
#             test_data_dm.x[:,0:4])**2+sklearn.metrics.pairwise_distances(dmsys.dmdata.x[train_sort][:,4:]
#             ,test_data_dm.x[:,4:])**2)
#         test_min_dist = numpy.min(test_dist,axis=0)

#         dm_outs=[]
#         for rs in xrange(nrealize):
#             clf.set_params(random_state=rs)
#             clf.fit(dmsys.dmdata.x[train_sort],dmsys.dmdata.y[train_sort])
#             y,dy2 = clf.predict(test_data_dm.x)
#             dm_outs.append(OneOutput(clf.feature_importances_,y,dy2))
#         dmoutput=Output(test_min_dist,train_cut,dm_outs)

#         pklfile=open(filename,'w')
#         pickle.dump([coloroutput,dmoutput],pklfile)
#     pklfile.close()


#     # w  = dmoutput.test_min_dist < dmoutput.train_cut
#     # plt.scatter(dmoutput.test_min_dist, dmoutput.outputs[0].y-test_data_dm.y)
#     # plt.xlim(0,dmoutput.train_cut*2)
#     # plt.show()

#     w  = coloroutput.test_min_dist < coloroutput.train_cut
#     plt.scatter(coloroutput.test_min_dist, coloroutput.outputs[0].y-test_data_dm.y)
#     plt.xlim(0,coloroutput.train_cut*2)
#     plt.show()

#     plt.scatter(dmoutput.outputs[0].dy2[w],dmoutput.outputs[0].y[w]-test_data_dm.y[w])
#     plt.show()

#     wef

#     if doplot:      
#         plt.clf()
#         figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         plt.clf()
#         figax= test_data_dm.plot(label='0.3',color='k',alpha=0.1,s=10,nsig=4)
#         test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.2),nsig=4, label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data_dm.plot(lambda x: passlimit(dmoutput,dmoutput.outputs[0],0.1),nsig=4 , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         plt.clf()
#         figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',alpha=1,s=10,figax=figax)
#         test_data.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',alpha=1,s=10,
#             figax=figax)
#         plt.show()

#         figax= test_data_dm.plot(label='0.3',color='k',nsig=4,alpha=0.1,s=10)
#         test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.2), label='0.2',color='b',nsig=4,alpha=1,s=10,figax=figax)
#         test_data_dm.plot(lambda x: passlimit(coloroutput,coloroutput.outputs[0],0.1) , label='0.1',color='r',nsig=4,alpha=1,s=10,
#             figax=figax)
#         plt.show()

#     c1,c2,c3,c4,c5, c6, c7 = meanuncertainties(test_data,coloroutput)
#     d1,d2,d3,d4,d5,d6,d7= meanuncertainties(test_data_dm,dmoutput)
#     # ind=0
#     # plt.clf()
#     # fig=plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.plot(c6[ind,:],label='c')
#     # ax.plot(d6[ind,:],label='d')
#     # plt.legend()
#     # plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.errorbar(c1,c2,yerr=c3,label='color',marker='o',color='b')
#     ax.errorbar(d1,d2,yerr=d3,label='dm',marker='o',color='r')
#     plt.legend()
#     plt.show()
#     fwe
#     # for a,b,c, d, e in zip(frac_include,mns.mean(axis=1),mns.std(axis=1),means,stds):
#     #     print "{:5.3f} {:7.4f} {:6.4f} {:6.4f} {:6.4f}".format(a,b,c, d, e)



#     # fwe
#     if doplot:
#         for index in xrange(8):
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.scatter(dmsys.dmdata.x[:,index],dmsys.dmdata.y)
#             ax.set_xlim((-20,20))
#         plt.show()
#     import matplotlib.collections
#     import matplotlib.lines
#     import matplotlib.patches
#     if doplot:
#         ## stuff to force the legend to have alpha=1
#         labs=['positive bias','negative bias','no bias']
#         # lines = [matplotlib.lines.Line2D([],[],color='r',marker='.',linestyle='None'),
#         # matplotlib.lines.Line2D([],[],color='b',marker='.',linestyle='None')
#         # ,matplotlib.lines.Line2D([],[],color='k',marker='.',linestyle='None')]
#         lines = [matplotlib.patches.Circle([], color='r'),matplotlib.patches.Circle([], color='b')
#             ,matplotlib.patches.Circle([], color='k')]

#         plt.clf()
#        # w = numpy.logical_and(split(train_data.x), numpy.abs(train_data.y) > x0[0])                    
#         figax= train_data.plot(lambda x: train_data.y > x0[0] , label='positive bias',color='r',alpha=0.02)
#         train_data.plot(lambda x: train_data.y < -x0[0] , label='negative bias',color='b',figax=figax,alpha=0.02)
#         train_data.plot(lambda x: numpy.abs(train_data.y) <= x0[0] , label='no bias',color='k',
#             figax=figax,alpha=0.02)

#         for ax in figax[1]:
#             for a in ax:
#                 a.legend(lines, labs,prop={'size':6})

#         plt.savefig('../results/colorspace.png')

#         plt.clf()
#         figax = dmsys.dmdata.plot(lambda x: dmsys.dmdata.y > x0[0] ,
#             label='positive bias',color='r',nsig=4,alpha=0.01)
#         dmsys.dmdata.plot(lambda x: dmsys.dmdata.y < -x0[0] ,
#             label='negative bias',color='b',figax=figax,nsig=4,alpha=0.01)
#         dmsys.dmdata.plot(lambda x: numpy.abs(dmsys.dmdata.y) <= x0[0] ,
#             label='no bias',color='k',figax=figax,nsig=4,alpha=0.01)

#         for ax in figax[1]:
#             for a in ax:
#                 a.legend(lines, labs,prop={'size':3})
#         plt.savefig('../results/dmspace.png')
     




#     # clf = objective_dm.clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)
#     # test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
#     # y_pred, y_var = clf.predict(test_data_dm.x)
#     # y_sig=numpy.sqrt(y_var)
#     # delta_y = y_pred-test_data.y
#     # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)
#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)
#     # ax1.legend()
#     # ax2.legend()
#     # plt.show()
    

#   # end old code
#     test_data_dm = Data(dmsys.coordinates(test_data.x,x0),test_data.y,test_data.z)
#     dists=numpy.sqrt(numpy.dot(dmsys.dmdata.x[:,0:4],numpy.transpose(test_data_dm.x[:,0:4]))**2 +
#         numpy.dot(dmsys.dmdata.x[:,4:],numpy.transpose(test_data_dm.x[:,4:]))**2)
#     dists = numpy.max(dists,axis=0)
#     w=dists < 1e15

#     # if True:
#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(111)
#     #     objective_dm.plot_scatter_external(ax,test_data_dm.x[w],test_data_dm.y[w],label='dm',color='b')
#     #     objective_color.plot_scatter_external(ax,test_data.x[w],test_data.y[w],label='color',color='r')
#     #     objective_dm.plot_scatter_external(ax,dmsys.dmdata.x[w],dmsys.dmdata.y[w],label='dm',color='b',marker='x')
#     #     objective_color.plot_scatter_external(ax,train_data.x[w],train_data.y[w],label='color',color='r',marker='x')

#     #     ax.legend()
#     #     plt.show()
#     #     wefwe

#     if True:
#         plt.clf()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
# #        ax.scatter(objective_dm.frac_include,objective_dm.fvals,label='dm',color='b')
# #        ax.scatter(objective_color.frac_include,objective_color.fvals,label='color',color='r')
#         objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b',marker='x')
#         objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r',marker='x')
#         ax.legend()
#         plt.savefig('../results/sigmas.png')
#         wefwe


#     if doplot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         objective_dm.plot_scatter(ax,label='Train dm',color='b')
#         objective_color.plot_scatter(ax,label='Train color',color='r')
#         ax.legend()

#     if doplot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         objective_dm.plot_scatter_external(ax,test_data_dm.x,test_data_dm.y,label='dm',color='b')
#         objective_color.plot_scatter_external(ax,test_data.x,test_data.y,label='color',color='r')


#         ax.legend()
#         plt.show()

#     # clf = clf.fit(dmsys.dmdata.x,dmsys.dmdata.y)

#     # y_pred, y_var = clf.predict(dmsys.dmdata.x)
#     # y_sig=numpy.sqrt(y_var)

#     # delta_y = y_pred-dmsys.dmdata.y

#     # fig1 = plt.figure()

#     # ax1 = fig1.add_subplot(111)
#     # ax1.scatter(y_sig, delta_y, label='train')

#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # fig2 = plt.figure()
#     # ax2 = fig2.add_subplot(111)
#     # ax2.scatter(y_sig[predind[1:]],std,label='train')


#     # 
#     # y_pred, y_var = clf.predict(test_data_dm.x)
#     # y_sig=numpy.sqrt(y_var)

#     # delta_y = y_pred-test_data.y

#     # ax1.scatter(y_sig, delta_y, label='test',color='r',alpha=0.1)

#     # sin = numpy.argsort(y_sig)
#     # delta_y=delta_y[sin]
#     # y_sig=y_sig[sin]
#     # std=[]
#     # predind = numpy.arange(10,len(delta_y),10)
#     # for i in xrange(len(predind)-1):
#     #     std.append(delta_y[predind[i]:predind[i+1]].std())
#     # std=numpy.array(std)
#     # ax2.scatter(y_sig[predind[1:]],std,label='test',color='r',alpha=0.1)

#     # ax1.legend()
#     # ax2.legend()

#     # plt.show()

# def old():




#     dm=DiffusionMap(train_data,x0[1])



#     if os.path.isfile('dmsys.pkl'):
#         #print 'get pickle'
#         pklfile=open('dmsys.pkl','r')
#         dm,test_data_dm=pickle.load(pklfile)
#     else:
#         # the new coordinate system based on the training data
#         dm=DiffusionMap(train_data,x0[1])
#         dm.make_map()
#         test_data_dm=Data(dm.transform(test_data.x),test_data.y,test_data.z,
#             xlabel=[str(i) for i in xrange(dm.neigen)])
#         pklfile=open('dmsys.pkl','w')
#         pickle.dump([dm,test_data_dm],pklfile)
#     pklfile.close()


#     ## plots that show dm x in color space
#     if doplot:
#         for i in xrange(6):
#             crap = numpy.sort(dm.data_dm().x[:,i])
#             crap= crap[len(crap)*.1:len(crap)*.9]
#             sig = crap.std()
#             cm=matplotlib.cm.ScalarMappable(
#                 norm=matplotlib.colors.Normalize(vmin=crap[len(crap)/2]-5*sig,
#                     vmax=crap[len(crap)/2]+5*sig),cmap='Spectral')
#             cval=cm.to_rgba(dm.data_dm().x[:,i])
#             figax= train_data.plot(c=cval,alpha=0.3,s=20,cmap=cm)
#             figax[0].suptitle(str(i))
#             plt.savefig('splits.'+str(i)+'.png')

#         figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,ndim=6)
#         dm.data_dm().plot(lambda x: outliers,
#             color='b',alpha=0.5,s=20,ndim=6,figax=figax)
#         plt.savefig('temp.png')
#         figax= dm.data_dm().plot(color='r',alpha=0.1,s=10,nsig=20,ndim=6)
#         dm.data_dm().plot(lambda x: outliers,
#             color='b',alpha=0.5,s=20,ndim=6,figax=figax)
#         plt.savefig('temp2.png')


#         # plt.clf()
#         # figax= test_data.plot(label='0.3',color='k',alpha=0.1,s=10)
#         # test_data.plot(lambda x: w, label='0.2',color='b',alpha=1,s=10,figax=figax)
#         # plt.show()

#     train_dist = sklearn.metrics.pairwise_distances(dm.data_dm().x,dm.data_dm().x)   
#     outlier_distances = train_dist[numpy.outer(outliers,outliers)]
#     outlier_distances = outlier_distances[outlier_distances !=0]
#     outlier_distances = numpy.sort(outlier_distances)
#     norm = outlier_distances[len(outlier_distances)*.05]

#     numpy.fill_diagonal(train_dist,train_dist.max()) #numpy.finfo('d').max)
#     train_min_dist = numpy.min(train_dist,axis=0)
#     closer  = train_min_dist < numpy.sort(train_min_dist)[train_data.ndata()*.95]
#     weight = numpy.exp(-(train_dist[outliers,:]/norm)**2).sum(axis=0)
#     if doplot:    
#         cm=matplotlib.cm.ScalarMappable(cmap='rainbow')
#         cval=cm.to_rgba(weight)
#         figax= train_data.plot(c=cval,alpha=0.1,s=20,cmap=cm)

#     weight_sort=numpy.sort(weight)

#     mn=[]
#     sd=[]
#     fracs=numpy.arange(0.01,0.5,0.05)
#     for frac in fracs:
#         w=weight < weight_sort[frac*train_data.ndata()]
#         w=numpy.logical_and(w,closer)
#         mn.append(train_data.y[w].mean())
#         sd.append(train_data.y[w].std())
#     plt.scatter(fracs,mn,c='b',label='mean')
#     plt.scatter(fracs,sd,c='r',label='std')
#     plt.show()

#     wfe
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

