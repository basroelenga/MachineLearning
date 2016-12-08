#!/usr/bin/env python

import numpy as np
from scipy import ndimage

a = np.zeros((6,6), dtype=np.float32)
a[2:4, 2:4] = 1
a[4, 4] = 1
a[:2, :3] = 2.5
a[0, 5] = 1
print a
print type(a)
print type(a[0,0])

label_a,num = ndimage.label(a)

print label_a

loc = ndimage.find_objects(label_a)

print loc
print len(loc)
print loc[0]
print a[loc[0]]
print a[loc[1]]
print a[loc[2]]
