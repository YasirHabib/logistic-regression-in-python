# Section 4, Lecture 30

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
N = 50
D = 50

#X = np.random.random((N, D))
#X = np.random.random((N, D)) - 0.5				# continuous uniform distribution centered around 0 from -0.5 to +0.5
X = (np.random.random((N, D)) - 0.5) * 10		# continuous uniform distribution centered around 0 from -5 to +5

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

# generate Y - add noise with variance 0.5
z = np.dot(X, true_w)
Target = np.round(sigmoid(z) + (np.sqrt(0.5) * np.random.randn(N)))

# perform gradient descent to find w
costs = [] # keep track of squared error cost
w = np.sqrt(1/D) * np.random.randn(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0

for t in range(5000):
	z = np.dot(X, w)
	predictions = sigmoid(z)
	xT = np.transpose(X)
	w = w - learning_rate * (np.dot(xT, predictions - Target) + l1 * np.sign(w))	# where dj/dw = np.dot(xT, Predictions - Target) + l1 * np.sign(w)
	
	# find and store the cost
	cost = -np.mean(Target * np.log(predictions) + (1 - Target) * np.log(1 - predictions)) + np.mean(l1 * np.abs(w))
	costs.append(cost)
	
# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot our w vs true w
plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()