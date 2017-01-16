'''
Created on Dec 23, 2016

@author: roelenga
'''

from __future__ import division

from sklearn.neural_network import MLPClassifier
import numpy as np

# Load in the data
data = np.loadtxt('dataset1.csv', delimiter=',')
data_names = np.loadtxt('named_dataset1.csv', delimiter=',')

print(data_names)

# Create a MLP object
mlp = MLPClassifier(hidden_layer_sizes=10, solver='lbfgs', learning_rate_init=0.01,max_iter=500)

# Fit our data, create a training set
training_set = np.ones(len(data))
mlp.fit(data, training_set)

print mlp.score(data, training_set)