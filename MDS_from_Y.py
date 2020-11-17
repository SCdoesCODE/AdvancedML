"""

MDS is very similar to PCA. But instead of converting
correlations in the data into a 2D graph, we convert
distances among the samples into a 2D graph. 

So we have to calculate the pair-wise distances between
all samples. But how do we calculate these distances?
One commong way is the euclidean distance, which is
the squared difference of all features summed up and then
the squared root of that summation. And then we apply
MDS to get the 2D graph. But the thing is that using
the euclidean distance will give the same 2D graph as 
in PCA. Meaning that clustering based on 
minimizing the linear distance is the
same as maximizing the linear correlations.

So therefore we use some other way to measure distances. 

Before doing MDS we want to normalize the data between 0 and 1 e.g..

"""
import numpy as np
from numpy import loadtxt
from eigen_decomposition import *

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
Y = []
for row in lines :
    Y.append([int(i) for i in row])

Y = np.transpose(np.array(Y))

#Y is the original n-dimensional representation of the data
S = np.dot(Y.T,Y)

V, lamda = eigen_decomp(S)

n = Y.shape[1]
k = 2

#new k_dimensional representation of data

X = np.dot(np.eye(k,n),np.dot(np.sqrt(lamda),V.T))


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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

x_points = X[0, :]
y_points = X[1, :]

texts = []
for idx,i in enumerate(names):
    x = x_points[idx]
    y = y_points[idx]
    plt.plot(x,y,"o",c = colormap[idx])
    plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=8)
    
    print(plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=8))


plt.show()

