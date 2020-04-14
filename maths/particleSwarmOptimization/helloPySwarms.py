# Import modules
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer
import matplotlib.pyplot as plt
import numpy as np

##########################
# Optimizer
##########################

# Set optimizer parameters
'''c1: cognitive parameter, c2: social parameter, w: inertia parameter (hyperparameters)'''
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
x_max = 10 * np.ones(2); x_min = -1 * x_max
bounds = (x_min, x_max)

# Set function to be optimised (from the list of default with pyswarms)
'''Function list:
- Ackley's, ackley  ;   Beale, beale    Booth, booth;    Bukin's No 6, bukin6
- Cross-in-Tray, crossintray;   Easom, easom    ;   Eggholder, eggholder    ;   Goldstein, goldstein
- Himmelblau's, himmelblau  ;   Holder Table, holdertable   ;   Levi, levi  ;   Matyas, matyas
- Rastrigin, rastrigin  ;   Rosenbrock, rosenbrock  ;   Schaffer No 2, schaffer2    ;     Sphere, sphere
- Three Hump Camel, threehump'''
functionName = 'crossintray'
functionList = {
    'ackley': {'limits': [(-4,4), (-4,4), (0,14)]},
    'crossintray': {'limits': [(-4,4), (-4,4), (-3,0)]},
    'sphere': {'limits': [(-1,1), (-1,1), (-0.1,1)]}}
# function = getattr(fx, functionName)
# limits = functionList[functionName]['limits']

# ---------------------------------
# Custom function
# ---------------------------------

# Custom sphere function
def sphere(inputs):
    x = inputs[:,0]; y = inputs[:,1]
    # return 20*(x + y) * np.exp(-6.*(x*x+y*y))
    return x**2 + y**2# -20*(np.cos(x[:,0]*3)*np.cos(x[:,1]*3))

# Create optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

# Run optimizer
function = sphere
cost, pos = optimizer.optimize(function, 1000)
limits = [(-4,4), (-4,4), (0,16)]


##########################
# Animate
##########################

# Constants
m = Mesher(func=function, limits=limits[0:2]) # get the sphere function's mesh (for better plots)
d = Designer(limits=limits, label=['x-axis', 'y-axis', 'z-axis']) # Adjust figure limits

# Animate in 3D
pos_history_3d = m.compute_history_3d(optimizer.pos_history) # preprocessing
animation3d = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0,0,0))
plt.show()

# # Animate swarm in 2D (contour plot)
# plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))
# plt.show()





# # Plot the cost
# plot_cost_history(optimizer.cost_history)
# plt.show()

# def cosCos(inputs):
#     x,y = inputs
#     return np.cos(x)*np.cos(y)
# function = lambda x : returnFitness(x, cosCos)
# from scipy.stats import multivariate_normal
# np.sum( -multivariate_normal.pdf(inputs, mean=2.5, cov=0.5))
# def function(x):
#     returnArray = np.zeros(shape=x.shape[0])
#     for i, inputs in enumerate(x):
#         returnArray[i] = np.sum( -multivariate_normal.pdf(inputs, mean=2.5, cov=0.5))
#     return returnArray
# function = lambda x : -multivariate_normal.pdf(x, mean=2.5, cov=0.5) # norm.pdf(x, loc=0, scale=1)

# # Set function to return fitness for a given function
# def returnFitness(x, fitnessFunction):
#     fitness = np.zeros(shape=x.shape[0])
#     for i, inputs in enumerate(x):
#         fitness[i] = fitnessFunction(inputs)
#     return fitness
# # cos(x)cos(y)
# def cosCos(inputs):
#     x,y = inputs
#     return np.sin(x*4)*np.cos(y*4)
# function = lambda x : returnFitness(x, cosCos)

# # See final result
# plt.figure()
# xRange = np.linspace(-.5,.5,20)
# yRange = np.linspace(-.5,.5,20)
# z = np.zeros(shape=[len(xRange),len(yRange)])
# for xIndx, x in enumerate(xRange):
#     for yIndx, y in enumerate(yRange):
#         z[xIndx,yIndx] = cosCos([x,y])
# plt.imshow(z)
# finalState = optimizer.pos_history[-1]
# plt.scatter(finalState[:,0], finalState[:,1])
# plt.scatter(bestResult[0],bestResult[1], color='k')
# plt.show()