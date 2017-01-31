#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import division

from sklearn.model_selection import train_test_split

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

from lasagne.nonlinearities import softmax
from lasagne.updates import adam, sgd, nesterov_momentum
from lasagne.layers import get_all_params

from nolearn.lasagne import PrintLayerInfo

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_loss

import matplotlib.pyplot as plt

import scipy.misc as scim
import lasagne as ls
import numpy as np
import pickle as pk
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

def create_neural_net(epochs=100, solver=sgd, hidden_layer_number=500, dropout=0.2, output_layers=3):
    
    layer_list = []
    
    # Input layer first
    layer_list.append(('input', InputLayer))
    
    layer_list.append(('conv1', Conv2DLayer))
    layer_list.append(('pool1', MaxPool2DLayer))
    layer_list.append(('conv2', Conv2DLayer))
    layer_list.append(('pool2', MaxPool2DLayer))
    layer_list.append(('conv3', Conv2DLayer))
    layer_list.append(('conv4', Conv2DLayer))
    layer_list.append(('pool3', MaxPool2DLayer))
    layer_list.append(('drop1', DropoutLayer))
    
    # Add the rest of the layers    
    layer_list.append(('hidden1', DenseLayer))
    layer_list.append(('hidden2', DenseLayer))
    layer_list.append(('drop2', DropoutLayer))
    layer_list.append(('output', DenseLayer))
                
    # Create the neural net
    neural_net = NeuralNet(
                     
        layers=layer_list,
                     
        # input layer
        input_shape=(None, x_train.shape[1], x_train.shape[2], x_train.shape[3]),
        
        # layer convolution and pooling
        conv1_num_filters=32, conv1_filter_size=(7, 7),
        pool1_pool_size=(2, 2),        
        conv2_num_filters=64, conv2_filter_size=(5, 5), 
        pool2_pool_size=(2, 2),       
        conv3_num_filters=128, conv3_filter_size=(4, 4),
        conv4_num_filters=128, conv4_filter_size=(3, 3),
        pool3_pool_size=(2, 2),
        
        drop1_p=dropout,

        # hidden layers
        hidden1_num_units=hidden_layer_number,
        hidden2_num_units=hidden_layer_number,
        drop2_p=dropout + 0.2, 
        # output layer
        output_nonlinearity=ls.nonlinearities.softmax,
        output_num_units=output_layers,
    
        # Optimization
        update=solver,
        update_learning_rate=0.0002,
                                  
        max_epochs=epochs,
    
        verbose=1,
    )
    
    return neural_net

def get_data(name, data):

    # Get the raw data from the jpeg files.
    data_list = []
    clas_list = []
    
    for i in range(0, len(name)):
        
        # Temporary data list
        temp_data_list = []
        
        # Prepare correct input shape for convolutional neural nets (color dim, x dim, y dim)
        for j in range(0, 3):
            
            single_color_channel = []
            
            for k in range(0, 50):
                for n in range(0, 50):
                    c = data[i][k][n][j]
                    single_color_channel.append(c)
            
            # Reshape the single color channel
            single_color_channel_reshaped = np.reshape(single_color_channel, (50, 50))
            temp_data_list.append(single_color_channel_reshaped)
        
        # Add it to the data list
        data_list.append(temp_data_list)
        
        # Get the image classification from the filename.
        temp_class = name[i].split("_")
        temp_class_list = []
            
        for j in range(0, len(temp_class)):
            
            if(j == 0): continue
            if(j == len(temp_class) - 1):
                
                classification = temp_class[j].split(".")[0]
                temp_class_list.append(classification)
            else:
                
                classification = temp_class[j]
                temp_class_list.append(classification)
                    
        clas_list.append(temp_class_list)

    return data_list, clas_list

def save_network(network, name):
    
    print("Saving network: " + name)
    network_parameters = network.get_all_params_values()
    
    with open(name, 'wb') as handle:
        pk.dump(network_parameters, handle, protocol=pk.HIGHEST_PROTOCOL)

# Load the images to test
image_data, name_data = load_images_from_folder("/net/dataserver2/data/users/nobels/MachineLearning/galaxyzoo/images_augmentation")
print("Images loaded")

# Get good data
data_list, clas_list = get_data(name_data, image_data)
    
# Create a deep convolution network.
# Convert data
x_train = np.asarray(data_list)
y_train = np.asarray(np.transpose(clas_list)[0]).astype(np.uint8)

# Create the network
print("Creating networks")

network = create_neural_net(epochs=400, solver=nesterov_momentum, hidden_layer_number=2048, dropout=0.2)

# Train the neural net
network.fit(x_train, y_train)

# Save network
save_network(network, "mainNet.p")

# Create a second neural network to do secondary analysis
# Get the correct data
data_spiral = []
data_ellips = []

# Define y_trains
y_train_spiral = np.transpose(clas_list)[2]
y_train_ellipse = np.transpose(clas_list)[1]

y_train_spiral_data = []
y_train_ellips_data = []

for i in range(0, len(y_train)):
    
    if(y_train[i] == 0): 
        
        data_spiral.append(x_train[i])
        y_train_spiral_data.append(y_train_spiral[i])
        
    else: 
        
        data_ellips.append(x_train[i])
        y_train_ellips_data.append(y_train_ellipse[i])

data_spiral = np.asarray(data_spiral)
data_ellips = np.asarray(data_ellips)
    
# Detect more from spirals
network_spiral = create_neural_net(epochs=400, solver=nesterov_momentum, hidden_layer_number=2048, dropout=0.2, output_layers=2)
network_ellips = create_neural_net(epochs=400, solver=nesterov_momentum, hidden_layer_number=2048, dropout=0.2)

# Train these networks
print("Training spiral")
network_spiral.fit(data_spiral, np.asarray(y_train_spiral_data).astype(np.uint8))

print("Training ellipse")
print(np.shape(data_ellips), np.shape(y_train_ellips_data))
network_ellips.fit(data_ellips, np.asarray(y_train_ellips_data).astype(np.uint8))

# Save the networks
save_network(network_spiral, "spiralNet.p")
save_network(network_ellips, "ellipsNet.p")

# Testing the neural networks
print("Predicting")

# Load data
image_data, image_name = load_images_from_folder('/net/dataserver2/data/users/nobels/MachineLearning/galaxyzoo/1000imagesforbas')

# Convert data
data_list, clas_list = get_data(image_name, image_data)

output_list = []

output_list.append(image_name)
output_list.append(network.predict(data_list))
output_list.append(network_spiral.predict(data_list))
output_list.append(network_ellips.predict(data_list))

for i in range(0, len(output_list)):
    print(output_list[i])
    
np.savetxt("output", np.transpose(output_list))