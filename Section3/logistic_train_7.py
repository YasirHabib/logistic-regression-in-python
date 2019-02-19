# Section 3, Lecture 23

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from logistic_process_2 import get_binary_data

X, Target = get_binary_data()
X, Target = shuffle(X, Target)

X_train = X[:-100]				# Retain all except last 100
Target_train = Target[:-100]    # Retain all except last 100
X_test = X[-100:]				# Retain last 100
Target_test = Target[-100:]		# Retain last 100

# Determine the dimensionality of the input matrix X
D = len(X[0])
w = np.random.randn(D)
b = 0 							# This is the bias term so that's a scalar

# make predictions
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
def forward(X, w, b):
	z = np.dot(X, w) + b
	return sigmoid(z)

# calculate the accuracy
def classification_rate(Target, predictions):
	return np.mean(Target == predictions)				# Alternatively we can also use (np.sum(predictions == y_test) / N) as given in Classification notes

# calculate the cross-entropy error
def cross_entropy(Target, predictions):
	return -np.mean(Target * np.log(predictions) + (1 - Target) * np.log(1 - predictions))

train_costs = []
test_costs = []
learning_rate = 0.001

for t in range(10000):
	predictions_train = forward(X_train, w, b)
	predictions_test = forward(X_test, w, b)
	
	ctrain = cross_entropy(Target_train, predictions_train)
	ctest = cross_entropy(Target_test, predictions_test)
	
	train_costs = np.append(train_costs, ctrain)
	test_costs = np.append(test_costs, ctest)
	
	xT = np.transpose(X_train)
	w = w - learning_rate * (np.dot(xT, predictions_train - Target_train))
	b = b - learning_rate * np.sum(predictions_train - Target_train)
	
	if t % 1000 == 0:
		print(t, ctrain, ctest)
 		
print("Final train classification_rate:", classification_rate(Target_train, np.round(predictions_train)))
print("Final test classification_rate:", classification_rate(Target_test, np.round(predictions_test)))

plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()