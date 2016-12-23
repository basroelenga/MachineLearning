from __future__ import division
from PIL import Image
import numpy as np
import numpy.random as r
import scipy.ndimage as sci
import matplotlib.pyplot as plt
import time
from random import sample


# loading and saving functions ###################################

def load(name):
    image = Image.open('%d.jpg'%name)
    return image


def output(image,i,name): #,N for which N images later
    filename = '%d_'%name + str(i) + '.jpg'
    image.save(filename)
    return


#functions for processing images after transformations. ##########

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


# transformation functions #######################################

#rotates image on random range 0-360 degrees
def rotation(image):
    rand = r.uniform(low=0,high=360)
    new_image = image.rotate(rand)
    return new_image


#function is temporarily out of order
def translation(image):
    rand = r.randint(-4,4,size=2)
    c = rand[0]
    f = rand[1]
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, c, 0, 1, f))
    return new_image


#function that zooms between 0.7-1.3 times
def zoom(image):
    width, height = image.size
    rand = r.uniform(0.7,1.3)
    new_image = image.resize((int(width*rand),int(height*rand)), Image.ANTIALIAS)
    new_width, new_height = new_image.size
    a = new_width/2
    new_image = new_image.crop((a-212,a-212,a+212,a+212))
    return new_image


#function which flips an image randomly either up down or right left or both or not at all
def flip(image):
    rand = r.randint(0,4)
    if rand == 0: 
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif rand == 1:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif rand == 2:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT) 
    else:
        new_image = image
    return new_image    


#function may not be used, is tricky to do right.  Need to learn PCA.
def brightness(image):
    dim = image.shape
    alphas = r.uniform(0,1,size=3) #randoms distribution unknown?!
    for i in range(0,dim[0]):
        for j in range(0,dim[0]):
            cov = np.cov(image[i,j])
            eig_values,eig_vectors = np.linalg.eig(cov)
            t_eig_values = np.transpose(eig_values)
            a_t_eig_values = np.multiply(alphas,t_eig_values)
            addition = np.dot(eig_vectors,a_t_eig_values)
            newimage[i,j] += addition
    return new_image            


#function which will sequentially apply the transformations to each image.
#each transformation is applied to the image once and only once.
def augment(n,name):
    start = time.time()
    image = load(name)
    images_array = []
    for i in range(0,n):
        idx = [0,1,2,3]
        rands = sample(idx,len(idx))
        new_image = image
        for j in rands:
            new_image = functions[j](new_image)
        new_image = crop(new_image)
        new_image = shrink(new_image)
        output(new_image,i,name)
    stop = time.time()
    print "Completed ", n, " augmentations in ", stop-start, " seconds"
    return 
      

#function that is useful for diagnostics, replaces broken transforms for now
def nothing(image):
    return image


n = 200
functions = [rotation, translation, zoom, flip]                       #brightness]  
function_names = ['rotation', 'translation', 'zoom', 'flip']      #'brightness']
augment(n,107846)
