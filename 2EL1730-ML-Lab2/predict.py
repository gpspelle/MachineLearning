import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    # Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta. The threshold is set at 0.5
    m = X.shape[0] # number of training examples
    c = np.zeros(m) # predicted classes of training examples
    p = np.zeros(m) # logistic regression outputs of training examples
    x = range(m)
    f = lambda x: 1 if x > 0.5 else 0
    for i in x:
        p[i] = np.dot(theta, X[i, :])
        c[i] = f(p[i])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #               training set.









    # =============================================================
    return c

