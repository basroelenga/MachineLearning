from __future__ import division
from sklearn import datasets
import numpy as np
from random import sample
import scipy.misc as scim
import os


## load image
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = scim.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
        except IOError:
            pass
    return images, filenames


images, filenames = load_images_from_folder("/Users/users/versteeg/Documents/MSC/ML")


## flatten along z-axis (add rgb values)
## NB change path to path of images
def flatten_RGB():
    gray_images = []
    for i in range(len(images)):
        gray_img = np.sum(images[i], axis=2)
        gray_images.append(gray_img)
    return gray_images


## convert image to row in data set array
def create_datasets():
    named_dataset = []
    dataset = []
    gray_images = np.asarray(flatten_RGB())
    for i in range(len(gray_images)):
        data_row = np.asarray(np.reshape(gray_images[i], (len(gray_images[i])**2)))
        dataset.append(data_row)
        row = [filenames[i], data_row]
        row = np.asarray(row, dtype=object)
        named_dataset.append(row)
    return np.asarray(named_dataset), np.asarray(dataset)

## named_dataset contains filename on index 0, data on index 1
## dataset contains pure data (ready for ML)
named_dataset, dataset = create_datasets()

##TO DO: allow for saving of filenames or named_dataset

#np.savetxt("named_dataset1.txt", named_dataset, delimiter=",")
#np.savetxt("names.txt", filenames)
#np.savetxt("dataset1.csv", dataset, delimiter=",", fmt='%.3d')


