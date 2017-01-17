#!/usr/bin/env python
from __future__ import division

import numpy as np
import os
import glob

A = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(1,2,3))

names = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(0,))

classification = np.zeros( len(A[:,0]) )
for i in xrange(0,len(A[:,0])):
     classification[i] = np.argmax(A[i,0:3])
     #if classification[i]==2:
     #     print 'YOLO  we found a loser! Motherfucker!'
     
image_names=[]
variation_names=[]
totalnames = []
for file in glob.glob("galaxyzoo/images_augmentation/*.jpg"):
    name = file
    totalnames.append(str(name))
    name = name.split('/')[-1]
    tempname = name
    name = name.split('-')[0]
    name2 = tempname.split('-')[1]
    name2 = name2.split('.')[0]
    image_names.append(str(name))
    variation_names.append(str(name2))

clas = np.zeros(len(image_names))
j=0
for i in xrange(0,len(image_names)):
     if int(image_names[i]) == int(names[j]):
          clas[i] = classification[j]
     else:
          j+=1
          clas[i] = classification[j]
     
     os.rename(totalnames[i],'galaxyzoo/images_augmentation/'+str(image_names[i])+'-'+str(variation_names[i])+'_'+str(int(clas[i]))+'.jpg' )


