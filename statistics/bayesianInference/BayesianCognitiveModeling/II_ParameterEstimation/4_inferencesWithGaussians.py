import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
import sys; sys.path.append("..") # Import common functions
from pymc3Functions import renderGraphicalModel, plotPosteriorDistribution
# see https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3

# Section to run
''''''
sectionToRun = 1

# ------------------------------------------------------------
# 4.1 Inferring a mean and standard deviation
# ------------------------------------------------------------
"""
One of the most common inference problems involves assuming data following a Gaussian distribution, and inferring
the mean and standard deviation of this distribution from a sample of observed independent data. The code below
shows an example of this process.

Here, the prior used for μ is intended to be only weakly informative. That is, it is a prior intended to convey
little information about the mean, so that inference will be primarily dependent upon relevant data. It is a
Gaussian centered on zero, but with very low precision (i.e., very large variance), and gives prior probability
to a wide range of possible means for the data. When the goal is to estimate parameters, this sort of approach is
relatively non-controversial.

Setting priors for standard deviations (or variances, or precisions) is trickier, and certainly more controversial.
If there is any relevant information that helps put the data on scale, so that bounds can be set on reasonable
possibilities for the standard deviation, then setting a uniform over that range is advocated by Gelman (2006).
In this first example, we assume the data are all small enough that setting an upper bound of 10 on the standard
deviation covers all the possibilities.
"""

if sectionToRun == 1: # inferringMeanSTD

    # Note on conversion between sigma and tau
    '''tau = 1/sigma**2  is  sigma = sqrt(1/tau)'''
    import sympy as sym
    sigma,tau = sym.symbols('sigma, tau')
    exp = 1/sigma**2-tau # tau = 1/sigma
    sym.solve(exp,sigma) # Here is automated algebra to get equation for sigma
    sigmaFromtau = lambda tau : np.sqrt(1/tau)

    # Data
    x = stats.norm.rvs(size=1000, loc=1, scale=1)

    # Define model 
    with pm.Model() as model:
        # prior
        mu = pm.Normal('mu', mu=0, tau=.001) # Gaussian prior for mu, centered at 0 but with low precision (Note that tau = 1/sigma^2)
        sigma = pm.Uniform('sigma', lower=0, upper=10) # Uniform prior over sigma (with a specific range given for these values)
        # observed
        xi = pm.Normal('xi',mu=mu, tau=1/(sigma**2), observed=x)
        # inference
        trace = pm.sample()

    # Render model and plot posterior distribution
    plt.subplot(1,2,1)
    plotPosteriorDistribution(trace.get_values('mu'), show=False)
    plt.subplot(1,2,2)
    plotPosteriorDistribution(trace.get_values('sigma'))
    renderGraphicalModel(model)
    # pm.traceplot(trace[50:]);

# ------------------------------------------------------------
# 4.2 The seven scientists
# ------------------------------------------------------------
"""
"""