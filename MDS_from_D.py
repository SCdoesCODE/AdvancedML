import numpy as np
from numpy import loadtxt
from eigen_decomposition import *

lines = loadtxt("zoo_boolean.txt", comments="#",dtype = str, delimiter=",", unpack=False)
#turn strings into int
Y = []
for row in lines :
    Y.append([int(i) for i in row])

Y = np.transpose(np.array(Y))

n = Y.shape[1]

"""

So we have our data. We want to compute a distance
matrix. It will be nr_points*nr_points in size and
will illustrate each pair-wise distance in the data.




"""
feature_importance = loadtxt("attribute_weights.txt", comments="#",dtype = str, delimiter=",", unpack=False)
feature_importance =[float(i)*10 for i in feature_importance]

D = np.zeros((n,n))

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        #euclidean distance
        D[i][j] = np.linalg.norm((Y[:,i]-Y[:,j])*feature_importance)

"""

Double centering the data. 

"""

ones = np.dot(np.ones((n,n)),np.ones((n,n)).T)

S = -0.5*(D-(n**-1)*np.dot(D,ones)-(n**-2)*np.dot(ones,np.dot(D,ones)))

V, lamda = eigen_decomp(S)

k = 3

#new k_dimensional representation of data

X = np.dot(np.eye(k,n),np.dot(np.sqrt(lamda),V.T))



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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import Counter

x_points = X[0, :]
y_points = X[1, :]
z_points = X[2, :]


figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')



texts = []


from adjustText import adjust_text

for idx,i in enumerate(names):
    x = x_points[idx]
    y = y_points[idx]
  
    plt.plot(x,y,"o",c = colormap[idx])
    texts.append(plt.text(x , y*1.05 , i, fontsize=10))
 
    #print(plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=20))
    #plt.annotate(i,(x,y),(x, y))
    plt.tight_layout()
    
adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()
