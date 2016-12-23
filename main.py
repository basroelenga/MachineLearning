#!/usr/bin/env python
from __future__ import division

import numpy as np

import sys

# load 1 of the 9 
datafile = int(sys.argv[1])
direct = './datalists/starting%d.txt'%datafile

# load the data of the training data
datatraining = np.loadtxt(direct)

classification = np.loadtxt('galaxyzoo/training_solutions_rev1.csv', skiprows=1,delimiter=',')

print classification[:,1:]

# generate all the required 
for i in xrange(0,len(datatraining)):
     print '%6d.jpg'%datatraining[i]

     
