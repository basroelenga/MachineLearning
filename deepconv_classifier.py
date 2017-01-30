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

from nolearn.lasagne import PrintLayerInfo

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_loss

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

def create_neural_net(epochs=100, conv_filter_depth=16, conv_layers=2, solver=sgd, hidden_layer_number=500, dropout=0.2):
    
    layer_list = []
    
    # Input layer first
    layer_list.append(('input', InputLayer))
    
    # Create the convolutional layers for the neural net
    for i in range(0, conv_layers):
        
        id_conv = 'conv' + str(i + 1)
        id_pool = 'pool' + str(i + 1)
    
        layer_list.append((id_conv, Conv2DLayer))
        layer_list.append((id_pool, MaxPool2DLayer))
    
    # Add the rest of the layers    
    layer_list.append(('hidden1', DenseLayer))
    layer_list.append(('drop1', DropoutLayer))
    layer_list.append(('output', DenseLayer))
        
    print(layer_list)
        
    # Create the neural net
    neural_net = NeuralNet(
                     
        layers=layer_list,
                     
        # input layer
        input_shape=(None, x_train.shape[1], x_train.shape[2], x_train.shape[3]),
        # layer conv2d1
        conv1_num_filters=conv_filter_depth, conv1_filter_size=(3, 3),
        pool1_pool_size=(2, 2),
        
        conv2_num_filters=conv_filter_depth * 2, conv2_filter_size=(3, 3),
        pool2_pool_size=(2, 2),
        
        # Add all the layer properties
        #for i in range(0, layers):
        
        # hidden layers
        hidden1_num_units=hidden_layer_number,
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
image_data, name_data = load_images_from_folder("images_augmentation_1000")
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
    temp_class = name_data[i].split("_")
    
    for j in range(0, len(temp_class) - 1):
        temp_class_list = []
        
        if(j == len(temp_class) - 2):
            classification = temp_class[1 + j].split(".")[0]
    
        else:
            classification = temp_class[i + j]
            
        temp_class_list.append(classification)
            
    clas_list.append(temp_class_list)
    
# Create a deep convolution network.
# Split the data.
x_train, x_test, y_train, y_test = train_test_split(data_list,
                                                    np.transpose(clas_list)[0],
                                                    test_size=0.1)

# Convert data
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

y_train = np.asarray(y_train).astype(np.uint8)
y_test = np.asarray(y_test).astype(np.uint8)

# Create the network
network = create_neural_net(epochs=1500, conv_filter_depth=32, solver=adam, hidden_layer_number=500, dropout=0.2)

# Initialize the network
network.initialize()

layer_info = PrintLayerInfo()
layer_info(network)

# Fit the data to the neural net
network.fit(x_train, y_train)

# Create a second neural network to do secondary analysis
# Get the correct data
data_spiral = []
data_ellipse = []

for i in range(0, len(y_train)):
    if(y_train[i] == 0): data_spiral.append(x_train[i])
    else: data_ellipse.append(x_train[i])
    
# Detect more from spirals
network_spiral = create_neural_net(conv_filter_depth=64, epochs=100)