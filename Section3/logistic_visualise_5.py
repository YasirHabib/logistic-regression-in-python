# Section 3, Lecture 19
# visualizes the Bayes solution

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# center the first 50 points at (-2,-2)
#X[:50,:] = X[:50,:] - 2*np.ones((50,D)))
X[:50,0] = X[:50,0] - 2
X[:50,1] = X[:50,1] - 2

# center the last 50 points at (2, 2)
#X[50:,:] = X[50:,:] + 2*np.ones((50,D))
X[50:,0] = X[50:,0] + 2
X[50:,1] = X[50:,1] + 2

# labels: first 50 are 0, last 50 are 1
Target = np.array([0]*50 + [1]*50)

# add bias term
X = np.column_stack([X, np.ones(N)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# get the closed-form solution

# wT = (meu1T - meu0T)(inverse of covariance matrix)
meu1 = np.array([2, 2])
meu1T = np.transpose(meu1)

meu0 = np.array([-2, -2])
meu0T = np.transpose(meu0)
# we know the covariance matrix is below. It has 1's in diagonal since the variance of both gaussians is a 1 because they are generated by
# np.random.randn. It has 0's in off-diagonals since they are uncorrelated & independent.
cov = np.array([[1, 0], [0, 1]])
cov_inv = np.linalg.inv(cov)
wT = np.dot((meu1T - meu0T), cov_inv)

# b = 0.5 * (meu0T inverse of covariance matrix)(meu0) - 0.5 * (meu1T inverse of covariance matrix) (meu1) - ln(alpha / 1-alpha)
alpha = 0.5
b = 0.5 * np.dot(np.dot(meu0T, cov_inv), meu0) - 0.5 * np.dot(np.dot(meu1T, cov_inv), meu1) - np.log(alpha / (1 - alpha))

wT = np.append(wT, [b])

# calculate the model output
z = np.dot(X, wT) + b
predictions = sigmoid(z)

plt.scatter(X[:,0], X[:,1], c = Target, s=100, alpha=0.5)
# Alternatively below is also correct
#plt.scatter(X[:50,0], X[:50,1], alpha=0.5)
#plt.scatter(X[50:,0], X[50:,1], alpha=0.5)
x_axis = np.linspace(-5, 5, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()