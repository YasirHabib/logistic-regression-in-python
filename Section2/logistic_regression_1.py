# demonstrates how to calculate the output of a logistic unit using numpy.
# the data X and weight matrix w are randomly generated from a standard normal distribution.


import numpy as np

N = 100
D = 2


X = np.random.randn(N, D)

# add bias term
X = np.column_stack([X, np.ones(N)])					# We can alternatively also use X = np.concatenate((X, np.ones((N, 1))), axis=1)

w = np.random.randn(D + 1)

z = np.dot(X, w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(z))