
# coding: utf-8

# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt


### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
print("Size of the data: ", data.shape)

#%%
# See data (five rows) using pandas tools
#print data.head(2)


### Prepare input to scikit and train and test cut

binary_data = data[np.logical_or(data['Cover_Type'] == 1, data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
print(np.unique(y))
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]

#%%
# Import cross validation tools from scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


#%%
### Train a single decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=100)

# Train the classifier and print training time
clf.fit(X_train, y_train)

#%%
# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = data['Cover_Type'].unique().astype(str).sort()
tree_pred = clf.predict(X_test)
print(classification_report(y_test, tree_pred, target_names=target_names))

#%%
# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tree_pred))


#===================================================================
#%%
### Train AdaBoost

# Your first exercise is to program AdaBoost.
# You can call *DecisionTreeClassifier* as above, 
# but you have to figure out how to pass the weight vector (for weighted classification) 
# to the *fit* function using the help pages of scikit-learn. At the end of 
# the loop, compute the training and test errors so the last section of the code can 
# plot the lerning curves. 
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

D = 2 # tree depth
T = 100 # number of trees
w = np.ones(X_train.shape[0]) / X_train.shape[0] # weight initialization
training_scores = np.zeros(X_train.shape[0]) # init scores with 0
test_scores     = np.zeros(X_test.shape[0])
 
# init errors
training_errors = []
test_errors = []
num_train = y_train.shape[0]
num_test = y_test.shape[0]
alphas = []

tree_pred_test = [] 
tree_pred_train = [] 
ada_pred_test = np.zeros(shape=(num_test)) 
ada_pred_train = np.zeros(shape=(num_train)) 

#===============================
for t in range(T):
    clf.fit(X_train, y_train, w)
    tree_pred_train.append(clf.predict(X_train))
    tree_pred_test.append(clf.predict(X_test))

    I = []
    for i in range(num_train):
        if tree_pred_train[-1][i] != y_train[i]:
            I.append(1)
        else:
            # TODO: if it explodes change it to -1
            I.append(0)

    # I contains which samples we did it right

    gama = 0
    soma = np.sum(w)
    for i in range(num_train):
        gama += w[i] * I[i]

    gama /= soma

    alphas.append(np.log((1 - gama)/gama))

    for i in range(num_train):
        w[i] = w[i] * np.exp(alphas[-1]*I[i])
    
    # Your code should go here
#===============================

for t in range(T):
    for i in range(num_train):
        ada_pred_train[i] += alphas[t] * tree_pred_train[t][i]
    for i in range(num_test):
        ada_pred_test[i] += alphas[t] * tree_pred_test[t][i]

ada_pred_test = np.sign(ada_pred_test)
ada_pred_train = np.sign(ada_pred_train)

print("*************************************")
print(accuracy_score(y_test, ada_pred_test))
print("*************************************")
print(accuracy_score(y_train, ada_pred_train))
print("*************************************")

#  Plot training and test error    
plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()

#===================================================================
#%%
### Optional part
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
    

#===============================
