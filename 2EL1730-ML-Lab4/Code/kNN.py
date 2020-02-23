import numpy as np
from matplotlib import pyplot as plt

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. Calculate the distance betweet y and each 
    #               row of X, find  the k closest observations and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance betweet two vectors A and B use
    #       use the np.linalg.norm() function.
    #


    # return the label of the test data

    num_train = X.shape[0]

    distances = dict()
    for i in range(num_train):
        distances[i] = np.linalg.norm(X[i]-y)

    #print(distances.sort())
    distance_sorted = {a: v for a, v in sorted(distances.items(), key=lambda arroz: arroz[1])}
    keys_min = list(distance_sorted.keys())
    keys_min = keys_min[:k]

    labels_closes = np.zeros(10)
    for i in keys_min:
        labels_closes[labels[i]] += 1

    # Show the first ten digits
    #fig = plt.figure('Closest digitis with L2 norm') 
    #for i in range(k):
    #    a = fig.add_subplot(2,5,i+1) 
    #    plt.imshow(X[keys_min[i],:].reshape(28,28), cmap=plt.cm.gray)
    #    plt.axis('off')

    #a = fig.add_subplot(2, 5, k+1)
    #plt.imshow(y.reshape(28, 28), cmap=plt.cm.gray)
    #plt.axis('off')
    #plt.show()

    return np.argmax(labels_closes)
