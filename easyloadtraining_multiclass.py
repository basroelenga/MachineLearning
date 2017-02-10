#!/usr/bin/env python
from __future__ import division

import numpy as np
import os
import glob

A = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(1,2,3))
B = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(16,17,18))
C = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(4,5))

names = np.loadtxt('galaxyzoo/training_solutions_rev1.csv',delimiter=',',usecols=(0,))

classificationA = np.zeros( len(A[:,0]) )
classificationB = np.zeros( len(B[:,0]) )
classificationC = np.zeros( len(C[:,0]) )

for i in xrange(0,len(A[:,0])):
     classificationA[i] = np.argmax(A[i,0:3])

for i in xrange(0,len(B[:,0])):
     classificationB[i] = np.argmax(B[i,0:3])
	 
for i in xrange(0,len(C[:,0])):
     classificationC[i] = np.argmax(C[i,0:2])
	 
	 
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

clasA = np.zeros(len(image_names))
clasB = np.zeros(len(image_names))
clasC = np.zeros(len(image_names))

j=0
for i in xrange(0,len(image_names)):
     if int(image_names[i]) == int(names[j]):
          clasA[i] = classificationA[j]
          clasB[i] = classificationB[j]
          clasC[i] = classificationC[j]
     else:
          j+=1
          clasA[i] = classificationA[j]
          clasB[i] = classificationB[j]
          clasC[i] = classificationC[j]  
		  
     os.rename(totalnames[i],'galaxyzoo/images_augmentation/'+str(image_names[i])+'-'+str(variation_names[i])+'_'+str(int(clasA[i]))+'_'+str(int(clasB[i]))+'_'+str(int(clasC[i]))+'.jpg' )


