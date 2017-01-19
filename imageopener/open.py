#!/usr/bin/env python

import numpy as np

from PIL import Image

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
#from nolearn.dbn import DBN
import numpy as np
#import cv2

image = Image.open('../galaxyzoo/images_test_rev1/102065.jpg')
image.show()
print image

with open('data.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')
