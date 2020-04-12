import pymc3 as pm
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Function to render graphical model
def renderGraphicalModel(model):
    pm.model_to_graphviz(model).render(filename='model', view=True, cleanup=True)

# Plot posterior distribution
def plotPosteriorDistribution(trace,x=np.linspace(0,1,100),show=True):
    posteriorDist = gaussian_kde(trace)
    plt.hist(trace, bins=100, normed=1, alpha=.3)
    plt.plot(x, posteriorDist(x), 'r') # distribution function
    if show==True: plt.show()