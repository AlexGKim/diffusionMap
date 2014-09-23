#!/usr/bin/env python
import numpy

seed=0
numpy.random.RandomState(seed)

def getcosmos():
    import pyfits
    file = "../data/cosmos-msdesi_filts.fits"
    hdulist = pyfits.open(file)
    return hdulist[1]

def setsplitter(ndata,  val_frac, opt_frac, n_vote):
    #ndata = data.header['NAXIS2']
    valind =  numpy.arange(0,ndata)
    numpy.random.shuffle(valind)
    cutind = numpy.round(ndata*val_frac)
    restind = valind[cutind:]
    valind = valind[0:cutind]

    restinds =  numpy.array_split(restind,n_vote)

    train = []
    for restind in restinds:
      cutind = numpy.round(len(restind)*opt_frac)
      op = restind[0:cutind]
      tr = restind[cutind:]
      train.append([op,tr])

    return valind, train
   
def main():
    #get data
    alldata = getcosmos()

    #id which are target and which are not
    targetind = alldata.data['O2'] >= 5e-17
    nontargetind = numpy.where(numpy.logical_not(targetind))[0]
    targetind = numpy.where(targetind)[0]
    split = [targetind, nontargetind]

    #make validation, training samples
    val_frac=0.1; opt_frac=0.1;  n_vote=10
    validate, train =setsplitter(len(split[0]), val_frac, opt_frac, n_vote)

    for i in xrange(1,len(split)):
      v, t = setsplitter(len(split[i]), val_frac, opt_frac, n_vote)
      validate=numpy.append(validate,v)
      for j in xrange(len(train)):
        train[j][0]=numpy.append(train[j][0],t[j][0])
        train[j][1]=numpy.append(train[j][1],t[j][1])


main()
