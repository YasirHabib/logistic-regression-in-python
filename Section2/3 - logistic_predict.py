import numpy as np
import pandas as pd

from logistic_process import get_binary_data

X, Y = get_binary_data()

# Determine the dimensionality of the input matrix X
D = len(X[0])

w = np.random.randn(D)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))