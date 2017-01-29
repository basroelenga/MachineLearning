#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import division

from sklearn.model_selection import train_test_split

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

from lasagne.nonlinearities import softmax
from lasagne.updates import adam, sgd
from lasagne.layers import get_all_params

from nolearn.lasagne.visualize import plot_loss

from nolearn.lasagne import PrintLayerInfo

from nolearn.lasagne import NeuralNet


import matplotlib.pyplot as plt

import scipy.misc as scim
import lasagne as ls
import numpy as np
import time
import sys
import os

t1 = time.time()
# Load in the data.
# load image.
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

def create_neural_net(epochs=100, solver=sgd, hidden_layer_number=500, dropout=0.2):
    
    # Create the neural net
    neural_net = NeuralNet(
                     
        layers=[
            ('input', InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('hidden4', DenseLayer),
            ('drop1', DropoutLayer),
            ('output', DenseLayer),
            ],
                     
        # input layer
        input_shape=(None, x_train.shape[1], x_train.shape[2], x_train.shape[3]),
        # layer conv2d1
        conv1_num_filters=16, conv1_filter_size=(3, 3),
        pool1_pool_size=(2, 2),
        
        conv2_num_filters=32, conv2_filter_size=(3, 3),
        pool2_pool_size=(2, 2),
        
        # hidden layers
        hidden4_num_units=hidden_layer_number,
        drop1_p=dropout, 
        # output layer
        output_nonlinearity=ls.nonlinearities.softmax,
        output_num_units=2,
    
        # Optimization
        update=solver,
        update_learning_rate=0.0002,
                                     
        max_epochs=epochs,
    
        verbose=1,
    )
    
    return neural_net

# Load the images to test
image_data, name_data = load_images_from_folder("/net/dataserver2/data/users/nobels/MachineLearning/galaxyzoo/images_augmentation")
print("Images loaded")

# Get the raw data from the jpeg files.
data_list = []
clas_list = []

for i in range(0, len(name_data)):
    
    # Temporary data list
    temp_data_list = []
    
    # Prepare correct input shape for convolutional neural nets (color dim, x dim, y dim)
    for j in range(0, 3):
        
        single_color_channel = []
        
        for k in range(0, 50):
            for n in range(0, 50):
                c = image_data[i][k][n][j]
                single_color_channel.append(c)
        
        # Reshape the single color channel
        single_color_channel_reshaped = np.reshape(single_color_channel, (50, 50))
        temp_data_list.append(single_color_channel_reshaped)
    
    # Add it to the data list
    data_list.append(temp_data_list)
    
    # Get the image classification from the filename.
    classification = name_data[i].split("_")[1].split(".")[0]
    clas_list.append(classification)
    
# Create a deep convolution network.
# Split the data.
x_train, x_test, y_train, y_test = train_test_split(data_list,
                                                    clas_list,
                                                    test_size=0.25)

# Convert data
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

y_train = np.asarray(y_train).astype(np.uint8)
y_test = np.asarray(y_test).astype(np.uint8)

print(np.shape(x_train))

# Create the network
network = create_neural_net(epochs=100, solver=adam, hidden_layer_number=500, dropout=0.2)

# Initialize the network
network.initialize()

layer_info = PrintLayerInfo()
layer_info(network)

# Fit the data to the neural net
network.fit(x_train, y_train)

fig = plt.figure()
plot_loss(network)

plt.show()