import numpy as np
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    classLabels = np.unique(Y)  # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape  # dimensions of the dataset
    totalMean = np.mean(X, 0)  # total mean of the data

    Sw = within_covariance(X, Y, dim, totalMean)
    Sb = per_class_covariance(X, dim, totalMean)

    _, W = eigen_problem(Sb, Sw)

    X_lda = np.dot(W, X)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.


    # =============================================================

    return W, projected_centroid, X_lda

def within_covariance(X, Y, K, totalMean):

    ocurrences = []
    ocurrences.append(list([index for index, value in enumerate(Y) if value == 1]))
    ocurrences.append(list([index for index, value in enumerate(Y) if value == 2]))
    ocurrences.append(list([index for index, value in enumerate(Y) if value == 3]))

    m = np.zeros(3)
    m[0] = np.sum([X[i] for i in ocurrences[0]]) / len(ocurrences[0])
    m[1] = np.sum([X[i] for i in ocurrences[1]]) / len(ocurrences[1])
    m[2] = np.sum([X[i] for i in ocurrences[2]]) / len(ocurrences[2])

    S_w = np.zeros((13, 13))

    for j in range(3):
        for i in ocurrences[j]:
            # index i related to a ocurrence of class j in data X
            S_w += np.dot(X[i] - m[j], p.transpose(X[i] - m[j])) 

    print(S_w)
    exit(1)

    S_w[0] = [np.dot(X[j] - m_1, np.transpose(X[j] - m_1)) for j in ocurrences_1] 
    S_w[1] = [np.dot(X[j] - m_2, np.transpose(X[j] - m_2)) for j in ocurrences_2] 
    S_w[2] = [np.dot(X[j] - m_3, np.transpose(X[j] - m_3)) for j in ocurrences_3]

    return S_w

def per_class_covariance(X, K, totalMean):

    ocorrucens_of_1 = [index for index, value in enumerate(Y) if value == 1]
    ocorrucens_of_2 = [index for index, value in enumerate(Y) if value == 2]
    ocorrucens_of_3 = [index for index, value in enumerate(Y) if value == 3]

    m_1 = np.sum([X[i] for i in ocurrences_of_1]) / len(ocurrences_of_1)
    m_2 = np.sum([X[i] for i in ocurrences_of_2]) / len(ocurrences_of_2)
    m_3 = np.sum([X[i] for i in ocurrences_of_3]) / len(ocurrences_of_3)

    m = (m_1 + m_2 + m_3) / 3

    S_b = np.zeros((3, 3))

    S_b[0] = [np.dot(m_1 - m, np.transpose(m_1 - m)) for j in ocurrences_1] * len(ocurences_of_1) 
    S_b[1] = [np.dot(m_2 - m, np.transpose(m_2 - m)) for j in ocurrences_2] * len(ocurrences_of_2) 
    S_b[2] = [np.dot(m_3 - m, np.transpose(m_3 - m)) for j in ocurrences_3] * len(ocurrences_of_3) 

    return S_b

def eigen_problem(Sb, Sw):

    eigven, eigvec = scipy.linalg.eig(np.dot(np.inverse(Sw), Sb)) 

    eigven = eigven[:2]
    eigvec = eigvec[:, :2]
    return eigven, eigvec 
