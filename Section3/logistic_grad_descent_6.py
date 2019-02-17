# dj/dw = summation(from n = 1 to N)(Predictions_n - Target_n)X_n  -> When in vector form
# dj/dw = np.dot(xT, Predictions - Target) which is 100% similar to gradient descent for linear regression -> dj/dw = np.dot(xT, Ypred - Y)

# Section 3, Lecture 21/22

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

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = np.dot(X, w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

predictions = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(Target, Y):
	j = 0
	for x in range(N):
		#j = j + -(Target[x] * np.log(Y[x]) + (1 - Target[x]) * np.log(1 - Y[x]))
		if Target[x] == 1:
			j -= np.log(Y[x])
		else:
			j -= np.log(1 - Y[x])
	return j
	
# let's do gradient descent 100 times
learning_rate = 0.1
for t in range(100):
	if t % 10 == 0:
		print(cross_entropy(Target, predictions))
	
	xT = np.transpose(X)
	w = w - learning_rate * (np.dot(xT, predictions - Target))		# where dj/dw = np.dot(xT, Predictions - Target)
	z = np.dot(X, w)
	predictions = sigmoid(z)
	
print("Final w:", w)