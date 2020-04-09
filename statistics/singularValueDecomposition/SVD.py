import numpy as np

# --------------------------------------------------
# Mathematical Overview (what does SVD return?)
# --------------------------------------------------
# https://www.youtube.com/watch?v=xy3QyyhiuY4&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv&index=2

"""
Singular Value Decomposition (SVD) is a commonly used method for data-compression
and dimensionality reduction. In SVD, a raw data matrix X is decomposed such that:

    X = U * Σ * V^T     (where ^T means matrix transpose)

U is called: the left singular vectors
V is called: the right singular vectors
Σ is called: the singular values

Matrices U and V are unitary. This means that U * U^T = the identity matrix

Matrices U and V are also orthonormal, meaning that their columns create vectors
of unit length, which are orthogonal to each other.

The principles of identity, unitary, and orthonormal matrices are coded below:
"""

# Create an orthonormal, unitary matrices
a = 0.2
b = (1-a**2)**0.5
matrix = np.array([[a,b],[-b,a]])

# Show that the matrix is othonormal (i.e. composed of unit vectors)
vectorMag = np.linalg.norm
print([vectorMag(matrix[r,:]) for r in range(matrix.shape[0])])
print([vectorMag(matrix[:,c]) for c in range(matrix.shape[1])])
# return [1.0,1.0], [1.0,1.0]

# Show that the matrix is unitary
print(np.round(np.dot(matrix, matrix.T)))
# return [[1,0],[0,1]] (i.e. the identity matrix)
# print(np.identity(2)) = array([[1., 0.], [0., 1.]])

"""
Imagine the input data is a series of pictures (e.g. of faces). The input data matrix
might be constructed such that each column of the matrix contains data from a single
image, which each row of a column is the value of a single pixel from that image.
We can refer to the number of rows in the matrix (i.e. the number of pixels
in a single image) as n, and the number of columns (i.e. the number of images in total),
as m.

Using the notation above, it is important to state that the size of matrix U is n*n, while
matrix V is m*m.

The matrix Σ is a diagonal matrix, with all values equalling zero except on the diagonal.
These values (σ) along the diagonal are 1) non-negative and 2) hierarchically ordered
(i.e. in decreasing magnitude such that σ1 > σ2 > σ3 ... > σm >= 0).
"""

# What does each value mean?
"""
U : You can think of matrix U as containing orthogonal principle components of the original
data matrix X. In other words, the first column of U will contain a single unit vector
which describes a single contributor to variance in matrix X. The first column of U will
describe the most important component, followed by the second most important, third, and
so on. Note that each of these components has the same size as a single column in the input
matrix (e.g. a single image), meaning that the components can also be visualised as an
example of an input column (e.g. an 'eigenface' in a collection of facial images).

Σ : The exact declining importance of each component is found in the Σ, with σ1 giving
the importance of the first component, σ2 for the second, and so on. (Note from above
that values of Σ always have decreasing magnitude). This is important as it allows us to
discard components that contribute very little to the variance of the original matrix X.

V : Matrix V essentially describes how each of the principle components (in U), after
being scaled by their respective σ value, can be combined to create an original column
in the input matrix X. If the input matrix is a series of facial images, the column in
V describes the mixture of each principle component in U required to recreate the
corresponding column image in matrix X. If this input matrix contains data which
changes with time across columns, V can be thought of as a kind of 'eigen-timeseries',
describing how each of the principle components change over time.
"""







# SVD of Potjans & Diesmann connectivity matrix
potjansDiesmann = np.array([
    [0.101, 0.169, 0.044, 0.082, 0.032, 0.0, 0.008, 0.0],#, 0.0],
    [0.135, 0.137, 0.032, 0.052, 0.075, 0.0, 0.004, 0.0],#, 0.0],
    [0.008, 0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.0],#, 0.0983],
    [0.069, 0.003, 0.079, 0.160, 0.003, 0.0, 0.106, 0.0],#, 0.0619],
    [0.100, 0.062, 0.051, 0.006, 0.083, 0.373, 0.020, 0.0],#, 0.0],
    [0.055, 0.027, 0.026, 0.002, 0.060, 0.316, 0.009, 0.0],#, 0.0],
    [0.016, 0.007, 0.021, 0.017, 0.057, 0.020, 0.040, 0.225],#, 0.0512],
    [0.036, 0.001, 0.003, 0.001, 0.028, 0.008, 0.066, 0.144]])#, 0.0196]])

import matplotlib.pyplot as plt
u, s, vh = np.linalg.svd(potjansDiesmann, full_matrices=True)
sigComps = np.where(s>.1)
sigUs = u[:,sigComps[0]]
sigVs = vh[sigComps[0],:]
components = sigUs.shape[1]
f1, axes1 = plt.subplots(components, 1)
f2, axes2 = plt.subplots(components, 1)
for i in range(components):
    axes1[i].plot(sigUs[:,i]); axes1[i].set_ylim([-1, .6])
    axes2[i].plot(sigVs.T[:,i]); axes2[i].set_ylim([-1, .6])
plt.show()


