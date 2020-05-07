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

def clusterSamples(nOfClusters):
    data = []
    pointsPerCluster = int(nPoints/nOfClusters)
    clusterLocations = np.random.uniform(low=-50,high=50,size=(nOfClusters,2))
    for c in range(nOfClusters):
        data.append(clusterLocations[c,:]+np.random.randn(pointsPerCluster,2))
    data = np.reshape(np.array(data), (pointsPerCluster*nOfClusters,2))
    return data

def sampleGridWithHole():
    dots = []
    xRange = np.arange(-3,3.5,step=.5)
    x,y = np.meshgrid(xRange,xRange)
    x = np.reshape(x,(len(x)**2))
    y = np.reshape(y,(len(y)**2))
    for i in range(len(x)):
        curX, curY = x[i], y[i]
        vec = np.array([curX,curY])
        if np.linalg.norm(vec)>2:
            dots.append([curX, curY])
    return np.array(dots)


# -------------------------------------------------
# Create data and analyse topology
# -------------------------------------------------

# Create data
x = np.random.uniform(low=0,high=2*np.pi,size=nPoints)
data = circleSample() # circleSample(), sphereSample(), torusSample, clusterSamples(5), sampleGridWithHole()

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
In this script, you will see that, when the data form clusters, there are h0 dots that have significantly higher
death value than the diagonal line (indicating persistent homology, and the presence of clusters in the data). The
number of these dots should be equal to the number of clusters. Alternatively, when the data is a 2D ring, there is a
h1 dot that has a signficantly higher death value than the diagonal line (indicating persistent homology, and the
presence of a circular hole in the data). However, when the data is a sphere, there is an h2 dot that has a signficantly
higher death value than the diagonal line (indicating persistent homology, and the presence of a void (or cavity) hole in
the data). In contrast to both, when data is taken from a torus, there is evidence of both a b1 and b2 holes.

It is also interesting to note that a grid of dots with a hole in the middle behaves quite similarly to a ring. In other
words, they both reveal the presence of a h1 dot (and therefore a circular hole). However, it is an important reminder than,
just because you see an h1 dot, doesn't necessary mean that the data are best described by a ring. 
'''

# # 3d sphere
# def sphereSample():
#     vec = np.random.uniform(-1,1,3)
#     vec /= np.linalg.norm(vec)
#     return vec
# data = np.array([sphereSample() for i in range(100)])
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(data[:,0],data[:,1],data[:,2])
# plt.show()