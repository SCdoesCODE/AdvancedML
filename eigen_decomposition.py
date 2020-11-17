"""

Eigendecomposition is a form of matrix decomposition

One can think of matrices as linear transformation - some rotate our vectors and some rescale them. 
A special type of vector is the eigenvector which does not change the direction, it might scale it though. 
Basically we get a new vector with the same direction. This means that the initial vector v and the transformen
vector Av is in the same direction. The output vector is just a scaled version of the initial vector, the
scale factor is called lambda. 

A*v = lambda * v

Eigendecomposition can only be used for square matrices (SVD can be used for non-square matrices)

"""

import numpy as np



def eigen_decomp(A):

    #return eigenvalues and eigenvectors of A
    [eigenvalues,eigenvectors] = np.linalg.eig(A)

    V = eigenvectors



    diag_lambda = np.diag(eigenvalues)

    #if the matrix is symmetrix, we can take the transpose of V instead, usually written as A =Q * diag(lambda) * Q^T
    V_inv = np.linalg.inv(V)

    decomp = V.dot(diag_lambda).dot(V_inv)
    #print(decomp)
    return(V,diag_lambda)






