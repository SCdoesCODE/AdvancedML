"""

Singular value decomposition

Compared to Eigendecomposition A can be a non-square matrix and is decomposed into 3 matrices instead of 2. 

A = U*D*V^T
- m*n

U = Left singular vectors
- m*m
D = Singular values
- m*n
V = Right singular vectors
- n*n

U and V are orthogonal matrices meaning that U^T = U^-1 and V^T = V^-1
D is a diagonal matrix (all zero except the diagonal) - however, not necessarily a square matrix


A can be seen as a linear transformation. When we decompose it down to U,D and V they all represent one subprocess
of this transformation, namely rotation, then rescaling, then rotation again. Transformations associated with diagonal
matrices imply only a rescaling of each coordinate without rotation, meanwhile matrices that are not diagonal
can produce a rotation.

"""

import numpy as np

A = np.array([[3, 7], [5, 2]])

#finding U,D,V directly with numpy
U, D, V = np.linalg.svd(A)

"""

Finding U, D and V of A

U corresponds to the eigenvectors of A*A^T - the left singular values of A
V corresponds to the eigenvectors of A^T*A - the right singular values of A
D corresponds to the square roots of the eigenvalues A*A^T or A^T*A which are the same - nonzero singular values of A

"""

A = np.array([[7, 2], [3, 4], [5, 3]])

#finding U,D,V through eigenvectors and eigenvalues

_, eigenvectors = np.linalg.eig(A.dot(A.T))
U = eigenvectors
_,eigenvectors = np.linalg.eig(A.T.dot(A))
V = eigenvectors
eigenvalues,_ = np.linalg.eig(A.dot(A.T))
D = np.sqrt(eigenvalues)


"""
EXAMPLE : we have an image described pixel by pixel in our matrix A. We decompose it using SVD and get our U,V, and D. 
The singular values and vectors are ordered such that the first ones correspond to the most variance explained. 
These first singular vectors and values represent the principal elements of the image, which we can use to reconstruct 
it. We can reconstruct the image with a certain number of singular values. E.g. if we choose 2 singular values
we use only the 2 first columns of the U matrix, the first columns and rows of the D matrix and the 2 first rows of 
the V matrix. Our reconstructed image remains the same dimension as 
"""

#number of singular values
i = 2

reconstructed = np.matrix(U[:, :i]) * np.diag(D[:i]) * np.matrix(V[:i, :])