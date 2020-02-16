from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import spdiags 

# Parameters
N = 150
spotSize = int(4/2)
typeOfInterest = 'spirals'
getRandom = lambda low, high : np.random.uniform(low,high)
parameters = {
    'corals': {'Du': 0.16, 'Dv': 0.08, 'F': 0.060, 'K': 0.062 },
    'spirals': {'Du': 0.12, 'Dv': 0.08, 'F': 0.020, 'K': 0.050},
    'zebrafish': {'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'K': 0.060},
    'bacteria': {'Du': 0.14, 'Dv': 0.06, 'F': 0.035, 'K': 0.065},
    'random': {'Du': getRandom(.12,.18), 'Dv': getRandom(.04,.10), 'F': getRandom(.02,.08), 'K': getRandom(.04,.07)}
}

class GrayScott():
    """Class to solve Gray-Scott Reaction-Diffusion equation"""

    # Constructor
    def __init__(self, N, parameters):
        self.parameters = parameters
        self.N = N
        self.u = np.ones((N, N), dtype=np.float128) # set all molecule u to 1
        self.v = np.zeros((N, N), dtype=np.float128) # set all molecule v to 0
        self.laplacianResult = []

    # Function to inject some v into the initial system
    def initialise(self):
        """Setting up the initial condition"""
        N, halfPoint, injectSize = self.N, np.int(self.N/2), spotSize
        self.laplacian()
        # self.u += 0.02*np.random.random((N,N))
        # self.v += 0.02*np.random.random((N,N))
        lowInx, highInx = halfPoint-injectSize, halfPoint+injectSize
        self.u[lowInx:highInx, lowInx:highInx] = 0.50
        self.v[lowInx:highInx, lowInx:highInx] = 0.25
        return  
    
    # Laplacian
    def laplacian(self):
        """Construct a sparse matrix that applies the 5-point discretization"""
        N = self.N
        e = np.ones(N**2)
        e2 = ([1]*(N-1)+[0])*N
        e3 = ([0]+[1]*(N-1))*N
        A = spdiags([-4*e,e2,e3,e,e],[0,-1,1,-N,N],N**2,N**2)
        self.laplacianResult = A
        return

    # Integrate
    def integrate(self):
        """Integrate the resulting system of equations using the Euler method"""

        # evolve in time using Euler method
        Du, Dv, F, K = self.parameters['Du'], self.parameters['Dv'], self.parameters['F'], self.parameters['K']
        L = self.laplacianResult
        u = self.u.reshape((N*N))
        v = self.v.reshape((N*N))

        # Reaction-diffusion equation
        uvv = u*v*v
        u += (Du*L.dot(u) - uvv +  F *(1-u))
        v += (Dv*L.dot(v) + uvv - (F+K)*v  )
    
        # return results
        self.u = u
        self.v = v
        return

    def onClick(self, event):
        v = self.v.reshape((N, N))
        xPnt, yPnt = int(event.xdata), int(event.ydata)
        getRange = lambda pnt: range(pnt-spotSize, pnt+spotSize)
        for x in getRange(xPnt):
            for y in getRange(yPnt):
                v[y, x] = 1
        self.v = v.reshape((N*N))


# Setup solver
rdSolver = GrayScott(N, parameters[typeOfInterest])
rdSolver.initialise()

# Setup figure
fig = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k') # dpi=400,
fig.canvas.mpl_connect('button_press_event', rdSolver.onClick) 
sp =  fig.add_subplot(1, 1, 1)
display = sp.pcolormesh(rdSolver.u.reshape((N, N)), cmap='binary')
plt.axis('off')

# Define animation function
def update(frame):
    for i in range(5): # Integrate many times before drawing to screen
        rdSolver.integrate()
    display.set_array(rdSolver.u)
    # sp.pcolormesh(rdSolver.u.reshape((N, N)), cmap=plt.cm.RdBu)
    return display

# Animate
ani = FuncAnimation(fig, update, interval=1)
plt.show()