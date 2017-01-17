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


images, filenames = load_images_from_folder("/net/dataserver2/data/users/nobels/MachineLearning/galaxyzoo/images_augmentation")

## flatten along z-axis (add rgb values)
## NB change path to path of images
def flatten_RGB():
    gray_images = []
    for i in xrange(len(images)):
        gray_img = np.sum(images[i], axis=2)
        gray_images.append(gray_img)
    return gray_images


## convert image to row in data set array
def create_datasets():
    named_dataset = []
    dataset = []
    #gray_images = np.asarray(flatten_RGB())
    for i in xrange(len(images)):
        data_row = images[i]
        dataset.append(data_row)
        row = [filenames[i], data_row]
        row = np.asarray(row, dtype=object)
        named_dataset.append(row)
    return np.asarray(named_dataset), np.asarray(dataset)

## named_dataset contains filename on index 0, data on index 1
## dataset contains pure data (ready for ML)
#named_dataset, dataset = create_datasets()

##TO DO: allow for saving of filenames or named_dataset
# Create file
str_data_header = ["#Training data"]
np.savetxt("/net/dataserver2/data/users/nobels/MachineLearning/dataset.dat", str_data_header, fmt="%s")

# Open handle
f_handle = open("/net/dataserver2/data/users/nobels/MachineLearning/dataset.dat", 'a')
print(np.shape(images))

# Save all data
for i in range(0, len(images)):
    
    print(i / len(images))
    
    # Save the filename
    str_datapart_header = ["f_name, 0, 0", filenames[i] + ", ,"]
    np.savetxt(f_handle, str_datapart_header, fmt='%s')
    
    image_array = []
    
    for j in range(0, len(images[i])):
        for k in range(0, len(images[i][j])):
            image_array.append(images[i][j][k])
        
    # Save the data
    np.savetxt(f_handle, image_array, fmt='%i', delimiter=',')
    
    # End data tag
    str_datapart_end = ["end, , "]
    np.savetxt(f_handle, str_datapart_end, fmt='%s')