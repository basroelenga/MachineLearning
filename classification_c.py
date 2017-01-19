#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import division

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import scipy.misc as scim
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

image_data, name_data = load_images_from_folder("images_augmentation_1000")
print("Images loaded")

# Get the raw data from the jpeg files.
data_list = []
clas_list = []

for i in range(0, len(name_data)):
    
    temp_data_list = []
    
    # Get the image data.
    for j in range(0, len(image_data[i])):
        for k in range(0, len(image_data[i][j])):
            for n in range(0, len(image_data[i][j][k])):
                                
                temp_data_list.append(image_data[i][j][k][n])
    
    # Get the image classification from the filename.
    classification = name_data[i].split("_")[1].split(".")[0]
    clas_list.append(classification)
    
    data_list.append(temp_data_list)

# define number of layers.
layernumb = int(sys.argv[1]) 
print("Number of neurons in hidden layer: " ,layernumb)

# Use the machine learning from sci-kit learn.
mlp = MLPClassifier(hidden_layer_sizes=(layernumb), activation="tanh", solver="lbfgs", max_iter=1000)

# Fit the data.
X_train, X_test, y_train, y_test = train_test_split(data_list,
                                                    clas_list,
                                                    test_size=0.25)

#mlp.fit(X_train, y_train)

# Test the model by predicting on the rest of the test set.
#print ('score', mlp.score(X_test,y_test))

# Test for best number of hidden layers
neurons = np.linspace(10, 400, 40)
score_list = []

print(neurons)

for i in range(0, len(neurons)):
    
    print(i / len(neurons) * 100, "%")
    
    mlp = MLPClassifier(hidden_layer_sizes=(neurons[i]), max_iter=1000, solver='lbfgs')
    mlp.fit(X_train, y_train)

    score_list.append(mlp.score(X_test, y_test))

fig = plt.figure()
plt.plot(neurons, score_list)

plt.xlabel("neurons")
plt.ylabel("score")

t2 = time.time()
print ('total time = %1.4f seconds' %(t2-t1))

plt.show()