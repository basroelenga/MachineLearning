#!/usr/bin/env python
from __future__ import division

import numpy as np

datatest = np.loadtxt('test.txt')
datatraining = np.loadtxt('training.txt')

for i in xrange(0,len(datatraining)):
     print '%6d.jpg'%datatest[i]

     print 
