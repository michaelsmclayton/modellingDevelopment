import pymc3 as pm
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Function to render graphical model
def renderGraphicalModel(model):
    pm.model_to_graphviz(model).render(filename='model', view=True, cleanup=True)

# Plot posterior distribution
def plotPosteriorDistribution(trace,show=True):
    posteriorDist = gaussian_kde(trace)
    plt.hist(trace, bins=100, density=True, alpha=.3)
    x = np.linspace(np.min(trace), np.max(trace), 100)
    plt.plot(x, posteriorDist(x), 'r') # distribution function
    if show==True: plt.show()