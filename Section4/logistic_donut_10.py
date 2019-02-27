# logisitc regression classifier for the donut problem.

import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi * np.random.random(N//2)
x = R1 * np.cos(theta)
y = R1 * np.sin(theta)
X_inner = np.column_stack([x, y])                         # We can alternatively also use X_inner = np.concatenate([[x], [y]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi * np.random.random(N//2)
x = R2 * np.cos(theta)
y = R2 * np.sin(theta)
X_outer = np.column_stack([x, y])

X = np.vstack([X_inner, X_outer])    					  # We can alternatively also use X = np.concatenate([X_inner, X_outer])
Target = np.array([0]*500 + [1]*500)

plt.scatter(X[:,0], X[:, 1], c = Target)
plt.show()

# create a column of r = sqrt(x^2 + y^2)
r = []
for t in range(N):
	r.append(np.sqrt(X[t,0] ** 2 + X[t,1] ** 2))

r = np.column_stack([r])

# add a column of r & ones
X = np.column_stack([X, r, np.ones(N)])

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
learning_rate = 0.0001
L = 0.01
for t in range(5000):
	e = cross_entropy(Target, predictions)
	cost.append(e)
	if t % 100 == 0:
		print(e)
	
	xT = np.transpose(X)
	w = w - learning_rate * (np.dot(xT, predictions - Target) + L*w)
	
	# recalculate predictions
	z = np.dot(X, w)
	predictions = sigmoid(z)

plt.plot(cost, label="Cross-entropy")
plt.title("Cross-entropy")
plt.legend()
plt.show()

print("Final w:", w)	
# returns Final w: [ 6.56046432e-03 -8.45826018e-03  1.58944775e+00 -1.18097569e+01]
# The first two weights are for x & y and are almost zero which means the classification doesn't really depend on the x & y coordinates at all is what
# our model has found. But it definitely depends on the radius (third weight) & the bias (fourth weight). So if we put in small radius, we automatically
# have this negative bias & that pushes the classification towards zero. And if the radius is bigger, it pushes the classification towards one. And so
# that's how you can solve the donut problem using logistic regression.
print("Final classification rate:", np.mean(Target == np.round(predictions)))