#!/usr/bin/env python
from __future__ import divison

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes(10),solver='sgd',learning_rate_init=0.01,max_iter=500)

mlp.fit(X_train, y_train)

print mlp.score(X_test,y_test)
