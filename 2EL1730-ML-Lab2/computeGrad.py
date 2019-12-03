import numpy as np
from sigmoid import sigmoid

def computeGrad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.
    m = X.shape[0] # number of training examples
    grad = np.zeros(theta.shape) # initialize gradient
    num_theta = theta.shape[0]

    for j in range(num_theta):
        for i in range(m):
            grad[j] += (X[i][j]) * (sigmoid(np.dot(theta, X[i, :])) - y[i])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.








    # =============================================================
    return grad
