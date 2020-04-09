import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_Points = 100
n_PointsInSpace = 500
numberOfTransforms = 4
numberOfImages = 8

# ----------------------------------------------
# Define transformation function types
# ----------------------------------------------
rand = lambda : (np.random.rand()*2)-1
class Linear():
    def __init__(self):
        self.a, self.b, self.c, self.d = rand(), rand(), rand(), rand()
    def transform(self, px, py):
        return (self.a * px + np.sin(self.b * py), self.c * px + self.d * py)

class ComplexSquare():
    def transform(self, px, py):
        z = complex(px, py)
        z2 = np.sqrt(z)
        return z2.real, z2.imag
'''see Modeling and Rendering of Nonlinear Iterated Function Systems'''

class Mobius():
    def __init__(self):
        self.a, self.b, self.c, self.d = rand(), rand(), rand(), rand()
    def transform(self, px, py):
        z = complex(px, py)
        z2 = (self.a*z + self.b) / (self.c*z + self.d) 
        return z2.real, z2.imag


# ----------------------------------------------
# Use Iterated Function System to iteratively produce images
# ----------------------------------------------
plt.figure(figsize=(12,12)) # Initialise figure

# Iterate over number of images wanted
for image in range(numberOfImages):
    print('Generating image %s...' % (image+1))

    # Get transforms
    transforms = [np.random.choice([ComplexSquare(), Linear()]) for i in range(numberOfTransforms)]

    # Iterate through points in space
    points = []
    for i in range(n_PointsInSpace):
        # Select input coordinates
        px = rand()
        py = rand()

        # Iteratively apply transformation functions
        for j in range(n_Points):

            # Choose transform
            t = np.random.choice(transforms)

            # Apply transform (and save result)
            px,py = t.transform(px,py)
            points.append([px,py])

    # Plot collect of all transformed points
    pointsArray = np.array(points)
    plt.subplot(2,4,image+1)
    plt.scatter(pointsArray[:,0], pointsArray[:,1], c='black', s=.01)
    plt.axis('off')

plt.show()