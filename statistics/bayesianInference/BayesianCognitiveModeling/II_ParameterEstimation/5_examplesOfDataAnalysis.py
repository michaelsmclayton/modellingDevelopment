import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
from theano import tensor
import sys; sys.path.append("..") # Import common functions
from pymc3Functions import renderGraphicalModel, plotPosteriorDistribution
# see https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3

# Section to run
''' 1: pearsonCorrelation, 2: pearsonCorrelationWithUncertainty'''
sectionToRun = 1


# ------------------------------------------------------------
# 5.1 Pearson correlation
# ------------------------------------------------------------
"""
The Pearson product-moment correlation coefficient, usually denoted r, is a widely used measure of the relationship between two
variables. It ranges from −1, indicating a perfect negative linear relationship, to +1, indicating a perfect positive relationship.
A value of 0 indicates that there is no linear relationship. Usually the correlation r is reported as a single point estimate,
perhaps together with a frequentist significance test. However, rather than just having a single number to measure the correlation,
it can be better to have a posterior distribution for r, saying how likely each possible level of correlation was.

One way of doing this is shown below. Here, the observed data take the form xi (= x1,...,xi for the ith observation) and, following
the theory behind the correlation coefficient, are modeled as draws from a multivariate Gaussian distribution. The parameters of this
distribution are the means μ = (μ1,μ2) and standard deviations σ = (σ1,σ2) of the two variables, and the correlation coefficient r
that links them.
"""
if sectionToRun == 1:

    # Create linearly correlated dataset
    def createLinearData(α, σ):
        x = np.random.uniform(low=0, high=10, size=11)
        y = α * x + (σ*np.random.rand(len(x)))
        return np.vstack((x,y)).T
    y = createLinearData(α=-1, σ=3)
    # y = np.array([.8, 102, 1,98, .5,100, 0.9,105, .7,103, 
    #                0.4,110, 1.2,99, 1.4,87, 0.6,113, 1.1,89, 1.3,93]).reshape((11, 2)) # Data from book

    # Define model
    with pm.Model() as model:
        # prior
        r =  pm.Uniform('r', lower=-1, upper=1) # uniform prior over correlation coefficient values (which can vary between -1 and 1)
        μ = pm.Normal('μ', mu=0, tau=.001, shape=2) # mean of two variables is set to 0, with low precision
        σ1, σ2 = pm.Gamma('σ1', alpha=.001, beta=.001), pm.Gamma('σ2', alpha=.001, beta=.001) # uninformative priors for within-variable standard deviations
        # create covariance matrix, created from correlation coefficient (r) and variable standard deviations (σ1, σ2)
        cov = pm.Deterministic('cov', tensor.stacklists([[σ1**2, r*σ1*σ2], [r*σ1*σ2, σ2**2]]))
        """ ---- Note on covariance matrices ----
        covarianceMatrix(x,y) = [[σ(x,x), σ(x,y)], i.e. the diagonal contains informance on within-variables VARIANCE (i.e. std**2),
                                while [σ(y,x), σ(y,y)]]   values away from the diagonal line give COVARIANCE between variables"""    
        # observed (likelihood)
        xi = pm.MvNormal('xi', mu=μ, cov=cov, observed=y, shape=2) # multivariate normal distribution created from covariance matrix
        # inference
        trace = pm.sample(draws=100)

    # Plot results
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(y[:,0],y[:,1])
    plt.subplot(2,1,2)
    plotPosteriorDistribution(trace.get_values('r'), show=False)
    rValue = stats.pearsonr(y[:,0],y[:,1])[0]
    plt.plot((rValue,rValue), plt.gca().get_ylim()) # Plot calculated Pearson correlation coefficient

    # Plot covariance matrix posteriors
    postCov = trace.get_values('cov')
    plt.figure(); pltInx = 1
    for r in range(2):
        for c in range(2):
            plt.subplot(2,2,pltInx)
            plotPosteriorDistribution(postCov[:,r,c], show=False)
            pltInx+=1
    plt.show()

    # Plot graphical model
    renderGraphicalModel(model)


# ------------------------------------------------------------
# 5.2 Pearson correlation with uncertainty
# ------------------------------------------------------------
"""
"""