# An implementation of:
''' Stępień, C. (2009). An IFS-based method for modelling horns, seashells and other natural forms.
Computers & Graphics, 33(4), 576-581'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.transform import Rotation as R

# Transformation parameters
organismChoice = 'Markhor'
parameterChoices = {
    'GreaterKudu': {'alpha': 7, 'beta': 7, 'gamma': 0, 'zScale': .96, 'parameters': 48},
    'WildGoat': {'alpha': 2, 'beta': 16, 'gamma': 0, 'zScale': .85, 'parameters': 8},
    'WaterBuffalo': {'alpha': .8, 'beta': 1.5, 'gamma': 0, 'zScale': .98, 'parameters': 80},
    'Markhor': {'alpha': 9, 'beta': 9, 'gamma': 0, 'zScale': .98, 'parameters': 90},
    'StrombusGigas': {'alpha': 0, 'beta': 40, 'gamma': 0, 'zScale': .93, 'parameters': 48},
    'ChiroceusRamosus': {'alpha': 0, 'beta': 80, 'gamma': 0, 'zScale': .85, 'parameters': 25},
    'LongBeak': {'alpha': 0, 'beta': 2, 'gamma': 0, 'zScale': .8, 'parameters': 18},
    'ShortBeak': {'alpha': 0, 'beta': 7.5, 'gamma': 0, 'zScale': .8, 'parameters': 18}}
alpha, beta, gamma, zScale, iterations = list(parameterChoices[organismChoice].values())

# -------------------------------------------
# Cube parameters
# -------------------------------------------
cubeCoordinates = [ \
    [0, 0, 0], # near bottom left
    [1, 0, 0], # near bottom right
    [1, 1, 0], # far bottom right
    [0, 1, 0], # far bottom left
    [0, 0, 1], # near top left
    [1, 0, 1], # near top right
    [1, 1, 1], # far top right
    [0, 1, 1]] # far top left

def getCubeVerts(coords, removeTopAndBottom=True):
    verts = [ \
        [coords[0], coords[1], coords[2], coords[3]],
        [coords[4], coords[5], coords[6], coords[7]],
        [coords[0], coords[1], coords[5], coords[4]],  
        [coords[2], coords[3], coords[7], coords[6]], 
        [coords[1], coords[2], coords[6], coords[5]],
        [coords[4], coords[7], coords[3], coords[0]]]
    if removeTopAndBottom==True:
        return verts[2:]
    else:
        return verts


# -------------------------------------------
# Transformation functions
# -------------------------------------------

def scale(vector, zScale):
    scaleMatrix = [[1,0,0], [0,1,0], [0,0,zScale]]
    return np.matmul(vector, scaleMatrix) 

def rotate(vector, alpha, beta, gamma):
    r = R.from_euler('zxz', [alpha, beta, gamma], degrees=True)
    return r.apply(vector)

def translate(priorShape, currentShape):
    diff = priorShape[4:,:] - currentShape[0:4,:] # top of prior cube, minus bottom of current cube
    currentShape[0:4] += diff
    currentShape[4:] += diff
    return currentShape

# -------------------------------------------
# Iterative transformations
# -------------------------------------------

def get3DItem(coords, color='black', fillColor=None):
    if fillColor == None: fillColor = color
    pc = Poly3DCollection(getCubeVerts(coords), edgecolors=color, linewidths=0.1)
    pc.set_alpha(.1)
    pc.set_facecolor(fillColor)
    return pc

def applyTransformation(transCube):
    transCube = rotate(transCube, alpha, beta, gamma)
    transCube = scale(transCube, zScale)
    return transCube

def updateAxisLimits(axisLimits, coords):
    def getMinMax(coords,dim):
        return [np.min(coords[:,dim]), np.max(coords[:,dim])]
    for dim in range(3):
        curMin, curMax = getMinMax(coords,dim)
        if curMin < axisLimits[dim][0]:
            axisLimits[dim][0] = curMin
        if curMax > axisLimits[dim][1]:
            axisLimits[dim][1] = curMax
    return axisLimits

def run():

    # Initialise figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1,1,1))
    axisLimits = [[np.float('inf'),0] for i in range(3)] # Initialise axis min & max values for all 3 dimensions

    # Get initial cube element
    transCube = np.copy(cubeCoordinates)
    transCube = applyTransformation(transCube)
    transCube[0:4,:] = np.copy(cubeCoordinates)[0:4,:]

    # Loop over iterations
    for i in range(iterations):
        originalShape = transCube
        ax.add_collection3d(get3DItem(originalShape))
        transCube = applyTransformation(transCube)
        transCube = translate(originalShape, transCube)
        axisLimits = updateAxisLimits(axisLimits, transCube)

    # Set axis limits and plot
    getAxisLimits = lambda limits : np.max(np.abs(np.subtract(limits,0))) # get greatest absolute distance from 0
    if organismChoice in ['StrombusGigas', 'ChiroceusRamosus']:
        ax.set_xlim(axisLimits[0][0], axisLimits[0][1])
        ax.set_ylim(axisLimits[1][0], axisLimits[1][1])
    else: # Rotate around center
        maxXYLimit = np.max([getAxisLimits(axisLimits[0]), getAxisLimits(axisLimits[1])])
        ax.set_xlim(-maxXYLimit, maxXYLimit)
        ax.set_ylim(-maxXYLimit, maxXYLimit)
    ax.set_zlim(axisLimits[2][0], axisLimits[2][1])
    plt.axis('off')

    # Animation function
    rotationSpeed = 3
    def animate(i):
        ax.view_init(elev=0, azim=i*rotationSpeed)
        return ax

    # Animate
    anim = animation.FuncAnimation(fig, animate, interval=1, frames=10000)#int(360/rotationSpeed),)
    # anim.save('%s.gif' % (organismChoice), writer='imagemagick', fps=60)
    plt.show()

if __name__ == "__main__":
    run()
