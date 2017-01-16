#!/usr/bin/env python
from __future__ import division
from PIL import Image
import numpy as np
import numpy.random as r
import scipy.ndimage as sci
import scipy.misc as scimisc
import itertools
import matplotlib.pyplot as plt
import time
import glob
from random import sample
import time 

n = input("How many galaxies would you like augmented today? ")

# loading and saving functions ####################################

def load(image_name):
    image = Image.open(image_name)
    return image


def output(image,image_name,i): #,N for which N images later
    image_name = image_name[:6]
    image.save("galaxyzoo/images_augmentation/%d-%d.jpg"%(int(image_name),i)) 
    return


#functions for pre-processing images prior to transformations. ####

#crops the input 424*424 images to 207*207
def crop(image):
    image = image.crop((108,108,315,315))
    return image


#function to downsample the cropped images down to 50*50
def shrink(image):
    basewidth = 50
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    return image


# transformation functions ########################################

#rotates image on random range 0-360 degrees
def rotation(image):
    rand = r.uniform(low=0,high=360)
    new_image = image.rotate(rand)
    return new_image


#function is not out of order
def translation(image):
    rand = r.randint(-4,5,size=2)
    width, height = image.size
    rand[rand ==0] = np.random.choice([-4,-3,-2,-1,1,2,3,4])
    new_image = Image.new("RGB", (width+rand[0], height+rand[1]))
    new_image.paste(image, (rand[0], rand[1]))
    rx = np.sign(rand[0])*np.random.randint(0,abs(rand[0]))
    ry = np.sign(rand[1])*np.random.randint(0,abs(rand[1]))
    new_image_cropped = new_image.crop((rx, ry, 424+rx, 424+ry))
    return new_image_cropped


#function is temporarily out of order
def zoom(image):
    width, height = image.size
    rand = r.uniform(1.0,1.2)
    new_image = image.resize((int(width*rand),int(height*rand)), Image.ANTIALIAS)
    new_width, new_height = new_image.size
    dx = int((new_width-width)/2)
    new_image = new_image.crop((dx,dx,424+dx,424+dx))
    return new_image


#flips an image left or right or both
def flip(image):
    rand = np.random.randint(0,3)
    if rand == 0: 
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif rand == 1:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
    return new_image  


#function may not be used, is tricky to do right
def brightness(image):
    dim = image.shape
    alphas = r.uniform(0,1,size=3) #randoms distribution unknown??? PCA
    for i in range(0,dim[0]):
        for j in range(0,dim[0]):
            cov = np.cov(image[i,j])
            eig_values,eig_vectors = np.linalg.eig(cov)
            t_eig_values = np.transpose(eig_values)
            a_t_eig_values = np.multiply(alphas,t_eig_values)
            addition = np.dot(eig_vectors,a_t_eig_values)
            newimage[i,j] += addition
    return new_image            


#function which will sequentially apply the transformatins to each image
def combinations(image,image_name,n):
    images_array = []
    for i in range(0,n):
        idx = [0,1,2,3]
        rands = sample(idx,len(idx))
        new_image = image
        for j in rands:
            new_image = functions[j](new_image)
        new_image = crop(new_image)
        new_image = shrink(new_image)
        output(new_image,image_name, i)
    return 
      

#function that is useful for diagnostics, replaces broken transforms for now
def nothing(image):
    return image

image_names = []
for file in glob.glob("galaxyzoo/images_training/*.jpg"):
    name = file[-10:-4]
    image_names.append(str(name))
print len(image_names)

functions = [rotation, translation, zoom, flip,] #brightness]  
function_names = ['rotation', 'translation', 'zoom', 'flip'] #'brightness']

t1 = time.time()

for i in xrange(100):#len(image_names)):
    image_name = image_names[i]
    image = load('galaxyzoo/images_training/'+image_name+'.jpg')
    combinations(image,image_name,n)

t2 = time.time()
print t2-t1
