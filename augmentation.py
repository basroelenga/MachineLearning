from __future__ import division
from PIL import Image
import numpy as np
import numpy.random as r
import scipy.ndimage as sci
import scipy.misc as scimisc
import itertools
import matplotlib.pyplot as plt
import time
from random import sample

n = input("How many galaxies would you like augmented today? ")

# loading and saving functions ####################################

def load():
    image = Image.open("107846.jpg")
    return image


def output(image,i): #,N for which N images later
    image.save("test_{}.jpg".format(i)) 
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


#function is temporarily out of order
def translation(image):
    rand = r.randint(-4,4,size=2)
    idx = r.randint(0,2)
    new_image = sci.interpolation.shift(image,rand[idx])
    return new_image


#function is temporarily out of order
def zoom(image):
    width, height = image.size
    rand = r.uniform(1.0,1.1)
    new_image = image.resize((int(width*rand),int(height*rand)), Image.ANTIALIAS)
    new_width, new_height = new_image.size
    a = new_width/2
    new_image = new_image.crop((a-103,a-103,a+104,a+104))
    return new_image


#function is temporarily out of order
def flip(image):
    rand = r.binomial(1,0.5)
    if rand == 1: 
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
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
def combinations(image,n):
    images_array = []
    for i in range(0,n):
        idx = [0,1,2,3]
        rands = sample(idx,len(idx))
        new_image = image
        for j in rands:
            new_image = functions[j](new_image)
        new_image = crop(new_image)
        new_image = shrink(new_image)
        output(new_image,i)
    return 
      

#function that is useful for diagnostics, replaces broken transforms for now
def nothing(image):
    return image


functions = [rotation, nothing, nothing, flip,] #brightness]  
function_names = ['rotation', 'translation', 'zoom', 'flip'] #'brightness']

image = load()
combinations(image,n)
