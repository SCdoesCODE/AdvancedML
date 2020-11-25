from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from SVD import *


"""

Here we will use SVD to do PCA. Instead of computing the covariance matrix
we will center the data and then compute the left/right singular values and 
the square roots of the eigenvalues - given by U,D,V. 

from the book :

"From a numerical point of view, the SVD
of the sample is more robust because it works on the whole data set, whereas
EVD works only on the summarized information contained in the covariance
matrix."
"""



"""

Load the data

"""

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
X = []
for row in lines :
    X.append([int(i) for i in row])

n = len(X)

"""

Let's center the data around 0. This is because PCA is a regression model
without an intercept and so the first component will inevitably cross the origin.
This is already done for us if we do PCA via the covariance matrix, but
if we do it through SVD we need to make sure that the data is centered. 

"""

def centerData(X):
    X = X.copy()
    X -= np.mean(X, axis = 0)
    return X

X_centered = centerData(X)



"""

Use SVD on the centered data

"""

U,D,V = SVD(X_centered)




"""
Calculate the principal components

"""

new_dim = 2


X_new = U[:,:new_dim]*D[:new_dim]



#we need to transpose X because the dimensions are on the columns in this case
#X_new = PCs.T.dot(X_centered.T)
names = loadtxt("short_names.txt", comments="#",dtype = str, delimiter="\n", unpack=False)

colormap = loadtxt("last_elem.txt", comments="#",dtype = str, delimiter="\n", unpack=False)
#string to int
colormap = [int(i) for i in colormap]

for idx,i in enumerate(colormap):
    if i == 1:
        colormap[idx] = "b"
    if i == 2:
        colormap[idx] = "g"
    if i == 3:
        colormap[idx] = "r"
    if i == 4:
        colormap[idx] = "c"
    if i == 5:
        colormap[idx] = "m"
    if i == 6:
        colormap[idx] = "y"
    if i == 7:
        colormap[idx] = "k"

from matplotlib.pyplot import figure
figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

x_points = X_new.T[0, :]
y_points = X_new.T[1, :]

texts = []
for idx,i in enumerate(names):
    x = x_points[idx]
    y = y_points[idx]
    plt.plot(x,y,"o",c = colormap[idx])
    plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=8)
    

plt.show()


