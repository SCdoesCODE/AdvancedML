from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt


"""

Load the data

"""

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
X = []
for row in lines :
    X.append([int(i) for i in row])

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

We can now start looking for PCs. From previously we know that we are looking for
the D which maximizes trace(d^T*X^T*X*d) with the constraint d^t*d = 1.
Remember that we can find the maximum of this trace function by finding the 
eigenvectors of X^T*X.


"""

eigVals, eigVecs = np.linalg.eig(X_centered.T.dot(X_centered))

"""

These are the vectors which maximize our function and each column vector
has a corresponding eigenvalue. The vector associated with the largest 
eigenvalue tells us the direction associated with the largest variance. 

Let's sort these so that the vector associated with the largest variance
is first.

"""

max_vector = eigVecs[np.argmax(eigVals)]

#negating the array sorts it in descending order
idx = np.argsort(-eigVals)
sorted_vectors = eigVecs[idx]

"""

Now that we have found our D we will use the encoding function to
rotate our data. The goal of the rotation is to end up with a new
coordinate system where the data is uncorrelated and where the basis axes
gather all the variance. It is then possible to keep only a few of the axes
which is the purpose of dimensionality reduction. 

Remember that the encoding function is : c = D^T*X.
Our D matrix contains our calculated eigenvectors. 

We want to rotate the data in order to have the largest variance 
on one axis. We can choose to keep this dimension and still have a 
pretty good representation of the data. 

"""

#we need to transpose X because the dimensions are on the columns in this case
X_new = eigVecs.T.dot(X_centered.T)
names = loadtxt("names.txt", comments="#",dtype = str, delimiter="\n", unpack=False)

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

x_points = eigVecs.T.dot(X_centered.T)[0, :]
y_points = eigVecs.T.dot(X_centered.T)[1, :]

texts = []
for idx,i in enumerate(names):
    x = x_points[idx]
    y = y_points[idx]
    plt.plot(x,y,"o",c = colormap[idx])
    plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=8)
    


plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.show()