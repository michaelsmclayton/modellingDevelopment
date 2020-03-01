from __future__ import division
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import matplotlib.animation as animation
import numpy as np
from scipy.sparse import spdiags 

# Parameters
N = 250
reflectiveBorders = True
spotSize = int(2/2)
typeOfInterest = 'spirals'
getRandom = lambda low, high : np.random.uniform(low,high)
parameters = {
    'corals': {'Du': 0.16, 'Dv': 0.08, 'F': 0.060, 'K': 0.062 },
    'spirals': {'Du': 0.12, 'Dv': 0.08, 'F': 0.020, 'K': 0.050},
    'zebrafish': {'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'K': 0.060},
    'bacteria': {'Du': 0.14, 'Dv': 0.06, 'F': 0.035, 'K': 0.065},
    'random': {'Du': getRandom(.12,.18), 'Dv': getRandom(.04,.10), 'F': getRandom(.02,.08), 'K': getRandom(.04,.07)}
}
convMatrix = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]

class GrayScott():
    """Class to solve Gray-Scott Reaction-Diffusion equation"""

    # Constructor
    def __init__(self, N, parameters):
        self.parameters = parameters
        self.N = N
        self.u = np.ones((N, N), dtype=np.float128) # set all molecule u to 1
        self.v = np.zeros((N, N), dtype=np.float128) # set all molecule v to 0
        self.laplacianMatrix = []

    # Function to inject some v into the initial system
    def initialise(self):
        """Setting up the initial condition"""
        N, halfPoint = self.N, np.int(self.N/2)
        self.laplacian()
        # self.u += 0.02*np.random.random((N,N))
        # self.v += 0.02*np.random.random((N,N))
        lowInx, highInx = halfPoint-spotSize, halfPoint+spotSize
        self.u[lowInx:highInx, lowInx:highInx] = 0.0
        self.v[lowInx:highInx, lowInx:highInx] = 1.0
        return  
    
    # Laplacian
    def laplacian(self):
        """Construct a sparse matrix that applies the 5-point discretization"""

        # Get convolution matrix to get Laplacian
        def getLaplacianConvMatrix(N):
            e = np.ones(N**2)
            diagonal = convMatrix[1][1]*e
            if reflectiveBorders==True:
                left = ([convMatrix[1][0]]*(N-1)+[0]) * N
                right = ([0]+[convMatrix[1][2]]*(N-1)) * N
            else:
                left = convMatrix[1][0]*e
                right = convMatrix[1][2]*e
            top = convMatrix[0][1]*e
            bottom = convMatrix[2][1]*e
            fullConvMatrix = [diagonal, left, right, top, bottom]
            return fullConvMatrix

        # Return laplacian multiplication mtarix
        fullConvMatrix = getLaplacianConvMatrix(self.N)
        self.laplacianMatrix = spdiags(data=fullConvMatrix, diags=[0,-1,1,-N,N], m=N**2, n=N**2)
        return


    # Integrate
    def integrate(self):
        """Integrate the resulting system of equations using the Euler method"""

        # evolve in time using Euler method
        Du, Dv, F, K = self.parameters['Du'], self.parameters['Dv'], self.parameters['F'], self.parameters['K']
        L = self.laplacianMatrix
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

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

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
    return display

# Animate
ani = animation.FuncAnimation(fig, update, interval=1, frames=10000, repeat=False)
# ani.save('greyScott_%s.mp4' % (typeOfInterest), writer=writer) # may require 'brew install' of 'brew update' of ffmpeg'
plt.show()
