#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import division

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import numpy as np

# Load in the data
data = np.genfromtxt("/net/dataserver2/data/users/nobels/MachineLearning/dataset.dat", dtype=str, delimiter=',')

# This will contain the ID
id_list = []
class_list = []

# This will contain the data
data_list = []
temp_data_list = []

should_skip = False

# Format the data for the machine learning
for i in xrange(0, len(data)):
    
    if(should_skip): 
        
        should_skip = False
        continue
    
    if(data[i][0] == 'f_name'):

        name = data[i + 1][0].split("_")
        
        id_list.append(name[0])
        class_list.append(name[1].split(".")[0])

        should_skip = True
    
    elif(data[i][0] == 'end'):

        data_list.append(np.transpose(temp_data_list))
        temp_data_list = []
        
    else:
        
        temp_data_list.append(int(data[i]))
    
# Use the machine learning from sci-kit
mlp = MLPClassifier()

print(np.asarray(data_list).shape, np.asarray(id_list).shape)

# Fit the data
X_train, X_test, y_train, y_test = train_test_split(data_list,
                                                    class_list,
                                                    test_size=0.25,
                                                    random_state=3)

mlp.fit(X_train, y_train)

print(X_test[0])
print("============")
print(X_test[1])

# Test the model
print 'score', mlp.score(X_test,y_test)

x = mlp.predict(X_test)
y = 0

print(x, y_test)

for i in range(0, len(x)):
    if(x[i] == y_test[i]):
        y += 1

print(y / len(x))
        