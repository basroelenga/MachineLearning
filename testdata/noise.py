#!/usr/bin/env python
from __future__ import division

import numpy as np

from scipy import ndimage

from astropy.io import fits

from matplotlib.pyplot import imsave

hdulist = fits.open('frame-g-000094-1-0100.fits.bz2')

dataarray = hdulist[0].data.byteswap().newbyteorder()

original = dataarray

print type(dataarray)

print dataarray

stdv = np.std(dataarray)

print stdv

dataarray[dataarray>stdv]=0.00143625

print np.std(dataarray)
print np.mean(dataarray)

meannoise = 0.00143808
stdnoise = 0.0251767

print np.random.normal(np.mean(dataarray),np.std(dataarray),size=(256,256))

#dataarray[dataarray>stdv]=1.0

'''
print dataarray

np.savetxt('dataraw.txt',dataarray,fmt='%1.1e')

print np.max(dataarray)


label_a,num = ndimage.label(dataarray)

print label_a

locs = ndimage.find_objects(label_a)


print locs
print dataarray[locs[0]]
print len(locs), num
imsave('loc1.png',dataarray[locs[0]])

for i in xrange(0,len(locs)):
     storefile = original[locs[i]]
     hdu = fits.PrimaryHDU(storefile)
     hdu.writeto('./testnick/frame-g-000094-1-0100-%04d.fits' % i)
     #storefile.writeto('./testnick/frame-g-000094-1-0100'+str(i)+'.fits')

#print locs

#imsave('image.png',dataarray)
#imsave('image5.png',np.log10(1+1e4*dataarray))
'''
