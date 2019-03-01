# logisitc regression classifier for the XOR problem.

import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Target = np.array([0, 1, 1, 0])

# create a column of xy = x*y
xy = X[:, 0] * X[:, 1]

# add a column of xy & ones
X = np.column_stack([X, xy, np.ones(N)])

# randomly initialize the weights
w = np.random.randn(D + 2)

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
cost = []
learning_rate = 0.001
L = 0.01
for t in range(5000):
	e = cross_entropy(Target, predictions)
	cost.append(e)
	if t % 100 == 0:
		print(e)
	
	xT = np.transpose(X)
	# gradient descent weight udpate with regularization
	w = w - learning_rate * (np.dot(xT, predictions - Target) + L*w)
	
	# recalculate predictions
	z = np.dot(X, w)
	predictions = sigmoid(z)

plt.plot(cost, label="Cross-entropy")
plt.title("Cross-entropy")
plt.legend()
plt.show()

print("Final w:", w)
print("Final classification rate:", np.mean(Target == np.round(predictions)))