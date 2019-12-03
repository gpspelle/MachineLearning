import numpy as np
from sigmoid import sigmoid

def computeCost(theta, X, y):
    # Computes the cost using theta as the parameter
    # for logistic regression.
    m = X.shape[0] # number of training examples
    # X is an array with dimensions (x, y)
    # X.shape
    J = 0

    for i in range(m):
        J += y[i] * np.log(sigmoid(np.dot(theta, X[i, :])))
        J += (1 - y[i]) * np.log(1 - sigmoid(np.dot(theta, X[i, :])))

    J *= (-1/m)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment
    #               for more details).







    # =============================================================
    return J
