"""

Order the attributes in accordance with how well they predict the "type" feature

Idea:
 - only use one feature to train the model at a time
 - for this we need to split the data into training and testing (say 70% training)
 - calculate the training error on the data for each feature
 - we probably want to have some randomization in how we split the train/test, but then we also need several runs
 - so per feature we do 10 runs where we train and test (using cross validation) - the average testing error is saved for each feature
 - the features are then ranked and weighted according to their test accuracy

"""

from sklearn import svm
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


"""

Load the data

"""

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
X = []
for row in lines :
    X.append([int(i) for i in row])

X = np.array(X)

types = loadtxt("last_elem.txt", comments="#",dtype = str, delimiter="\n", unpack=False)
targets = np.array([int(i) for i in types])

accuracies = []
std = []


for col in range(X.shape[1]):
    for i in range(10):
        a = []
        s = []


        x = X[:,col].reshape(X.shape[0],1)

        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, x, targets, cv=10)

        a.append(scores.mean())
        s.append(scores.std())
        
        #print(("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
    
    accuracies.append(np.mean(a))
    std.append(np.mean(s))


weights = open("attribute_weights.txt", "w")
for a in accuracies:
    weights.write(str(a))
    weights.write("\n")
