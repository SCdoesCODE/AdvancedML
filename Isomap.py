"""

Nonlinear dimensionality reduction

Given some high dimensional data which possibly lies on a nonlinear manifold 
we reconstruct the the low-dimensional coordinates of the data which describe 
where the points lie on the manifold. 

Here Euclidean distance may be a poor measure of distance. Therefore dimensionality
reduction techniques like PCA cannot accurately describe the unlinear nature of data.
E.g. we want to account for the "swirl" in some data in a higher-dimensional space
even in the lower dimensional space.Instead geodesic distance might be a better at 
preserving the neighborhour relations in the manifold. 

However, calculating the geodesic distance is difficult without knowing
the exact manifold. A guiding principle is that if two points are close enough
the geodesic distance is approximately equal to the euclidean distance. 
For points that are further away we construct a sequence of short "hops"
between neighboring points. Isomap constructs a map for computing this. 

The Isomap algorithm uses the same core ideas as the MDS algorithm. 
There is one main difference - the way the distance matrix is constructed.
In MDS we use the euclidean distance to get the distance between two points.
In Isomap it is the weights of the shortest past in a point-graph. 
We then use an eigen-decomposition of this distance matrix (double-centered) to get our 
lower-dimensional embedding of the data. 

How to construct the point graph : we place an edge between two points
if the euclidean distance between them falls under a certain threshold
or between a point and its top k neighbours. 

So we start by adding an edge between each point and its nearest neighbor,
given by the euclidean distance. The weight of each edge is the euclidean
distance between the connected points. The distance between two points
is given by the weights of the shortest path between those points. And
this is how our distance matrix is computed. 

In the above example k=1, but we could also have other k, meaning that we place
more edges between points. 

"""

import numpy as np
from numpy import loadtxt
from eigen_decomposition import *

"""

LOAD DATA

"""

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
Y = []
for row in lines :
    Y.append([int(i) for i in row])

Y = np.transpose(np.array(Y))

#nr of points
n = Y.shape[1]


"""

CONSTRUCT POINT GRAPH
inf if no edge could be constructed

"""

k = 4

NG = np.ones((n,n)) * np.inf
P = np.ones((n,n))

PG = np.ones((n,k)) * np.inf
#this keeps track of which points(their indexes 0-100) are the nearest neighbors
PG_points = np.ones((n,k))


for i in range(n):
    for j in range(n):
        if i != j :
            NG[i][j] = np.linalg.norm(Y[:,i]-Y[:,j])
        P[i][j] = j


for i,row in enumerate(NG):
    idx = row.argsort()
    NG[i] = NG[i][idx]
    P[i] = P[i][idx]




PG = NG[:,:k]
PG_points = P[:,:k]



"""
for i in range(n):
    for j in range(n):
        dist = np.linalg.norm(Y[:,i]-Y[:,j])
        #dist will be zero when we compare it to itself and we do not want to include that
        #first part of or : update array if we find a new smallest value
        #second part of or : as long as we have inf values, we just want to update the array with any distance
        if(dist < np.min(PG[i]) and dist!=0) or (dist!=0 and np.inf in PG[i]):
            PG[i][np.argmax(PG[i])] = dist
            PG_points[i][np.argmax(PG[i])] = j


 """           

"""

COMPUTE DISTANCE MATRIX
using weights of shortest paths between each pair-wise points
"all pairs shortest path"

Using Floyd Warshall algorithm

Time complexity : O(n^3) due to the nested for loop where we update the distance matrix

"""

D = np.ones((n,n)) * np.inf
#the distance of a node to itself is zero
np.fill_diagonal(D,0)

#fill in the weights of the edges, only the k-nearest neighbors will be filled
for i in range(n):
    for j in range(n):
        if j in PG_points[i]:
            D[i][j] = PG[i][list(PG_points[i]).index(j)]

for k in range(n):
    for i in range(n):
        for j in range(n):
            #when we find a new shortest path  from i to j, we update the distance matrix
            if D[i][j] > D[i][k] + D[k][j]:
                D[i][j] = D[i][k] + D[k][j]
              
for i in range(n):
    for j in range(n):
        if D[i][j] == np.inf:
            D[i][j] = 10000000



"""

Now we have our distance matrix which we feed to MDS
First : double centering

"""

ones = np.dot(np.ones((n,n)),np.ones((n,n)).T)

S = -0.5*(D-(n**-1)*np.dot(D,ones)-(n**-2)*np.dot(ones,np.dot(D,ones)))

V, lamda = eigen_decomp(S)

k = 3

#new l_dimensional representation of data

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


#figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

x = [float(i) for i in X[0, :]]
y= [float(i) for i in X[1, :]]
z = [float(i) for i in X[2, :]]


from mpl_toolkits.mplot3d import Axes3D

from collections import Counter

def create_new_labels(three_dim = False):

    xi = [int(i) for i in x]
    yi= [int(i) for i in y]
    zi = [int(i) for i in z]

    if three_dim:
        #a is the axis we are concerned with, either y or z
        a = zi
        p = list(zip(xi,yi,zi))
    else:
        a = yi
        p = list(zip(xi,yi))
    
    

    distinct_points = list(Counter(p).keys())
    point_count = list(Counter(p).values())

    a_count = point_count

    a_new = np.zeros(n)

    for idx,i in enumerate(p):
        
        count_idx = distinct_points.index(i)
        total_count_for_i = point_count[count_idx]
        if total_count_for_i > 1:
            a_count[count_idx] -= 1
            a_new[idx] = a[idx] + 200*(a_count[count_idx])
        else :
            a_new[idx] = a[idx]

    
  

    return a_new


def plot_3d(labels = True):

    z_label = create_new_labels(three_dim = True)

    fig = plt.figure(figsize=(15, 8))


    ax = fig.add_subplot(111,projection = "3d")
    ax.scatter(x, y,z, color = colormap)

    if labels : 
        for i, txt in enumerate(names):
            ax.text(x[i],y[i],z_label[i], s =   '%s' % (str(txt)), bbox=dict(facecolor=colormap[i],alpha = 0.5),size=7, zorder=1,  
            color='k') 


    plt.savefig("image.png",bbox_inches='tight',dpi=100)

    plt.show()

def plot_2d(labels = True):
    y_label = create_new_labels()

    fig = plt.figure(figsize=(15, 8))


    ax = fig.add_subplot(111)
    ax.scatter(x, y, color = colormap)

    if labels :

        for i, txt in enumerate(names):
            ax.text(x[i],y_label[i], s =   '%s' % (str(txt)),size=7, zorder=1,  
            color='k') 


    plt.savefig("image.png",bbox_inches='tight',dpi=100)

    plt.show()


plot_2d()
plot_3d()
plot_3d(labels = False)
