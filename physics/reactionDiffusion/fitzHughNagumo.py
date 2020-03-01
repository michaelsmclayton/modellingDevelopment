import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import matplotlib.animation as animation

# Equation parameters
a = 0.00028
b = 0.005
k = -.005

# Simulation parameters
tau = .1
size = 100  # size of the 2D grid
dx = 2. / size  # space step
dt = .001  # time step

# Laplacian function
def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

# Initialise U and V
U = np.random.rand(size, size)
V = np.random.rand(size, size)

# Setup figure
fig = plt.figure()# , axes = plt.subplots(3, 3, figsize=(8, 8))
display = plt.imshow(U, cmap='binary', interpolation='bilinear', extent=[-1, 1, -1, 1])

# Animation function
def update_patterns(n):

    for i in range(100): # Iterate system 100 timess

        # We compute the Laplacian of u and v.
        deltaU = laplacian(U)
        deltaV = laplacian(V)

        # We take the values of u and v inside the grid.
        Uc = U[1:-1, 1:-1]
        Vc = V[1:-1, 1:-1]

        # We update the variables.
        U[1:-1, 1:-1], V[1:-1, 1:-1] = \
            Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k),\
            Vc + dt * (b * deltaV + Uc - Vc) / tau
        
    # Neumann conditions: derivatives at the edges are null.
    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]

    # Set display and return
    display.set_array(U)
    return display

# Animate
ani = animation.FuncAnimation(fig, update_patterns, interval=1, frames=10000, repeat=False)
plt.axis('off')
plt.show()

# see 'https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns'
