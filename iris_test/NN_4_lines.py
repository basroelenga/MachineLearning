import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
datatrain = pd.read_csv('iris.csv')

print type(datatrain)

# Change string value to numeric
datatrain.set_value(datatrain['species']=='setosa',['species'],0)
datatrain.set_value(datatrain['species']=='versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)

# Change dataframe to array
datatrain_array = datatrain.as_matrix()

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:4],
                                                    datatrain_array[:,4],
                                                    test_size=0.2)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print 'score', mlp.score(X_test,y_test)

sl = 5.8
sw = 4
pl = 1.2
pw = 0.2
data = [sl,sw,pl,pw]
prediction = mlp.predict(data)
if prediction == 0:
	print 'given', data, 'species is setosa'
elif prediction == 1:
	print 'given', data, 'species is versicolor'
if prediction == 2:
	print 'given', data, 'species is virginica'
