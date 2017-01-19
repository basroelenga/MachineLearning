#!/usr/bin/env python
from __future__ import division

import numpy as np

i = 1
j = 100

runs = [94,109,125,211,240,241,250,259,273,287,297,307,308,745,752,756,994, 1009,1011,1013,1022,1024,1033,1035,1037,1040,1043,1045,1056,1057,1122,1140, 1231,1239,1241,1302,1331,1345,1350,1356,1359]

#print len(runs)

for k in xrange(0,16):
     for i in xrange(1,7):
          for j in xrange(11,500):
               print str(runs[k])+'|'+str(i)+'|'+str(j)


