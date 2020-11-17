"""

Popular dimensionality reduction tool. Not feature selection, instead it creates a new set of dimensions
called principal components. These are ordered such that the first one is the dimension associated with the
largest variance. The PCs are also orthogonal meaning that each PC is decorrelated to the preceding one. 
Remember that orthogonal vectors have a 0 dot product, they are perpendicular to each other. 
This means that we can choose to keep only the first few PCs knowing that each PC is a linear combination 
of the original features/dimensions. 

An overview : we want to go from n to l dimensions. We need a function that does this, and also a function
that converts the l-dimensional data back to the n-dimensional data. 

So we have an encoder function f(x) = c and a decoder function g(c) = x

We have our initial data matrix X with m points : n*m - meaning each column is an n-dimensional point
Our embedded data matrix C with with m points : l*m - meaning each column is an l-dimensional point

Our encoding function f(X) = C transforms X to C
And our decoding function g(C) = X tranforms C back to X

There are some constraints to this encoding and decoding
 - the decoding function has to be a simple matrix multiplication 
 - g(C) = Dc ---- with dimensions n*l = n*l * l*l
 - the columns of D should be orthogonal
 - the columns of D must have unit norm (be orthonormal) - AKA have a length S


Let's start with one point. We will find the encoding function from the decoding function. We want 
to decode one point c. We want to minimize the error between the decoded data point and the actual data point. 
This means that we want to reduce the distance between x and g(c). As an indicator of this distance we use the squared L2-norm.

 - remember g(c) = Dc - an approximation of x

How to calculate the p-norm
 - calculate the absolute value of each element
 - take the power p of these absolute values
 - sum all these powered absolute values
 - take the pth root of this summed result 

Squared L2-norm removes that last step of taking the pth root. So basically abs(x-g(c)). Will give a vector with
some numbers. Take these to the power of 2 and then sum up. We want to find a c that minimizes this. 

Let's say y = g(c) - x

The L2-norm can also be calculated as y^T*y

Thus the equation that we want to minimize becomes (g(c) - x)^T(g(c) - x)

(g(c) - x)^T(g(c) - x) = {transpose respects addition} = (g(c)^T - x^T)(g(c) - x) = {distributive property} = 
x^T*x - x^T*g(c) - g(c)^T*x + g(c)^T*g(c) = {commutivity} = x^T*x - x^T*g(c) - x^T*g(c) + g(c)^T*g(c)
= x^T*x - 2*x^T*g(c) + g(c)^T*g(c)

The first term x^T*x does not have to do with c and because we want to minimize the expression with regards to
c we can just get rid of it. 

We simplify to : 

g(c)^T*g(c) - 2*x^T*g(c) 

Remember g(c) = Dc

(Dc)^T*Dc - 2*x^T*Dc = D^T*c^T*Dc - 2*x^T*Dc = c^T*D^T*D*c - 2*x^T*Dc
{D^T*D = I because D is orthogonal(if n is not equal to l and have unit norm columns)} 
= c^T*I*c - 2*x^T*Dc = c^T*c - 2*x^T*Dc

Our final expression which we want to minimize : c^T*c - 2*x^T*Dc
 - popular way of doing this is to use gradient descent, one problem is getting stuck in local minima.

Calculating the gradient of the function : We want to minimize through each dimension of c. We are looking
for a slope 0. So we want to calculating the derivative of the expression with respect to c and then set
it equal to zero. 

The derivative of the expression (and set to 0) : 

2c - 2*x^T*D = 0
c = x^T*D

To make the dimensions work

c = D^T*x

So our final encoding function it

c = D^T*x

to go back from c to x we use

g(c) = Dc


But how do we find D?

Recall that in PCA we want to change coordinate systems such that we maximize the variance along the first dimension.
This is equivalent to minimizing the error between data points and their reconstruction. 

D will be used for all points and to calculate it we will use the Frobenius norm of the errors which is equivalent to the 
L2-norm for matrices. Basically like unrolling the matrix into a vector and then taking the L2 norm of that. 

So we want the D which minimized the Frobenius norm, and the only constraint is that D^T*D = I 
(because Ds columns are orthogonal).

Finding the first principal component : we set l = 1. D will be n*1 -  a column vector which we can now call d. 

To calculate the reconstruction error we use the reconstruction function. r(x) = g(f(x))) = D*D^T*x
 - because recall g(c) = Dc  - decoding
 - and c = D^T*x   - encoding

We want the d which minimizes the reconstruction error. We take the L2-norm of x-r(x)
- r(x) = D*D^T*x
- since we are only looking for the first PC : r(x) = d*d^T*x
- so we take the L2-norm of X - X*d*d^T with the constraint that d^T*d = 1
- we use the trace operator to simplify the expression to minimize
- frobenius(X - X*d*d^T) = square(trace((X - X*d*d^T)(X - X*d*d^T)^T))
    - frobenius(A) = square(trace(A*A^T))
- due to the cycling property of traces and because we don't need terms that does not involve d
    we can simplify the expression to
    trace(d^T*X^T*X*d) with the constraint d^t*d = 1
    we want to find the d which maximized this expression

We can find the maximum of this trace function by finding the eigenvectors of X^T*X.
 - maximizing the variance of the components and minimizing the reconstruction error is equivalent
 - if we have centered our data around 0, X^T*X is the covariance matrix
 - the covariance matrix is an n*n matrix whose diagonal is the covariance of the corresponding dimensions
    and the other cells are the pair-wise covariances corresponding to two dimensions
 - we want to maximize variance and minimize covariance between dimensions (to make them decorrelated).
    This means that the ideal covariance matrix is a diagonal matrix with only the diagonal non-zero.
    Therefore the diagonalization of the covariance matrix will give us the optimal solution. 
 
"""

import numpy as np
import matplotlib.pyplot as plt



"""

Let's create some multi-dimensional data and correlate it

"""

np.random.seed(123)
x = 5*np.random.rand(100)
y = 2*x + 1 + np.random.randn(100)
#z = 5*y + np.random.rand(100)
#a = 2*z + np.random.randn(100)

x = x.reshape(100, 1)
y = y.reshape(100, 1)
#z = z.reshape(100, 1)
#a = a.reshape(100, 1)

X = np.hstack([x, y])


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

plt.plot(eigVecs.T.dot(X_centered.T)[0, :], eigVecs.T.dot(X_centered.T)[1, :], '*')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()