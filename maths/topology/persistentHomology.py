import numpy as np
import matplotlib.pylab as plt
from ripser import ripser # pip3 install scikit-tda
from persim import plot_diagrams
from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets

# Parameters
nPoints = 100

# -------------------------------------------------
# Define functions to create datasets
# -------------------------------------------------

def circleSample():
    data = np.array([np.cos(x),np.sin(x)])
    data += .15*np.random.randn(2,len(x))
    return data.T

def sphereSample():
    def sample_spherical(npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
    sphereSamples = sample_spherical(nPoints)
    sphereSamples += .15*np.random.randn(3,nPoints)
    return sphereSamples.T

def torusSample(r=1, R=3, x0=0, y0=0, z0=0):
    u = 2 * np.pi * np.random.rand(nPoints)
    v = 2 * np.pi * np.random.rand(nPoints)
    cosu = np.cos(u)
    sinu = np.sin(u)
    cosv = np.cos(v)
    sinv = np.sin(v)
    x = x0 + (R + r * cosu) * cosv
    y = y0 + (R + r * cosu) * sinv
    z = z0 + r * sinu
    return np.array([x, y, z]).T

# -------------------------------------------------
# Create data and analyse topology
# -------------------------------------------------

# Create data
x = np.random.uniform(low=0,high=2*np.pi,size=nPoints)
data = torusSample() # circleSample(), sphereSample(), torusSample

# Analyse topology
diagrams = ripser(data, maxdim=data.shape[1]-1)['dgms']

# Plot results
fig = plt.figure()
fig.set_figwidth(5); fig.set_figheight(15)
if data.shape[1]==3:
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(data[:,0],data[:,1],data[:,2])
else:
    ax1 = fig.add_subplot(211)
    ax1.scatter(data[:,0],data[:,1])
ax1.set_title('Data')
ax2 = fig.add_subplot(212)
plot_diagrams(diagrams, show=True, ax=ax2)
plt.show()

''' Note that this second graph shows the persistant of different n-dimensional homologies. These dimensions
can be referred to using Betti number:
    - b0 is the number of connected components
    - b1 is the number of one-dimensional or "circular" holes
    - b2 is the number of two-dimensional "voids" or "cavities"
In this script, you will see that, when the data is a 2D ring, there is a h1 dot that has a signficantly higher
death value than the diagonal line (indicating persistent homology, and the presence of a circular hole in the data).
However, when the data is a sphere, there is an h2 dot that has a signficantly higher death value than the diagonal
line (indicating persistent homology, and the presence of a void (or cavity) hole in the data). In contrast to both,
when data is taken from a torus, there is evidence of both a b1 and b2 holes 
'''