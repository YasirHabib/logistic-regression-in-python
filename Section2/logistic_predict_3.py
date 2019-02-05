import numpy as np
import pandas as pd

from logistic_process_2 import get_binary_data

X, Y = get_binary_data()

# Determine the dimensionality of the input matrix X
D = len(X[0])

w = np.random.randn(D)
b = 0 # This is the bias term so that's a scalar

def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
def forward(X, w, b):
	z = np.dot(X, w) + b
	return sigmoid(z)

P_Y_given_X = forward(X, w, b)
predictions = np.round(P_Y_given_X)			# Refer to Section2 Lecture 7 to understand this.

def classification_rate(Y, P):
	return np.mean(Y == P)					# Alternatively we can also use (np.sum(predictions == y_test) / N) as given in Classification notes
	
score = classification_rate(Y, predictions)
print("Score:", round(score*100,2), "%")