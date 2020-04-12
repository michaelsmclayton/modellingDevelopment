import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
import sys; sys.path.append("..") # Import common functions
from pymc3Functions import renderGraphicalModel, plotPosteriorDistribution
# see https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3

# Section to run
'''1: inferringMeanSTD, 2: sevenScientists, 3: repeatedMeasurementIQ'''
sectionToRun = 3

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
        μ = pm.Normal('μ', mu=0, tau=.001) # Gaussian prior for mu, centered at 0 but with low precision (Note that tau = 1/sigma^2)
        σ = pm.Uniform('σ', lower=0, upper=10) # Uniform prior over sigma (with a specific range given for these values)
        # observed
        xi = pm.Normal('xi',mu=μ, tau=1/(σ**2), observed=x)
        # inference
        trace = pm.sample()

    # Render model and plot posterior distribution
    plt.subplot(1,2,1)
    plotPosteriorDistribution(trace.get_values('μ'), show=False)
    plt.subplot(1,2,2)
    plotPosteriorDistribution(trace.get_values('σ'))
    renderGraphicalModel(model)
    # pm.traceplot(trace[50:]);


# ------------------------------------------------------------
# 4.2 The seven scientists
# ------------------------------------------------------------
"""
In this problem, seven scientists with wildly-differing experimental skills all make a measurement of the same quantity.
They get the answers x (see below). Intuitively, it seems clear that the first two scientists are pretty inept measurers,
and that the true value of the quantity is probably just a bit below 10. The main problem is to find the posterior
distribution over the measured quantity, telling us what we can infer from the measurement. Our secondary problem,
however, is to infer something about the measurement skills of the seven scientists.

One way to solve this problem is to assume that all the scientists offer measurements that come from a Gaussian
distribution. As the scientists all measure the same thing, the mean of these distributions should all be the same
(i.e. μ). However, as each scientist has different measurment abilities, the standard deviations of these distributions
(i.e. σ; or error of their measurements, will be different between scientists. In the code below, this principle is
illustrated with the fact that λ (or the distribution precision) is different for each of the scientists (i.e. with a
shape of n).
"""
if sectionToRun == 2: # sevenScientists

    # Measurement data (i.e. from each of the seven scientists)
    x = np.array([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])
    n = len(x)

    # Define model
    with pm.Model() as model: 
        # prior
        μ = pm.Normal('μ', mu=0, tau=.001)
        λ = pm.Gamma('λ', alpha=.01, beta=.01, shape=(n)) # Note here that setting shape to n means that λ is estimated seperately for each scientist
        σ = pm.Deterministic('σ', 1/np.sqrt(λ))
        # observed
        xi = pm.Normal('xi', mu=μ, tau=λ, observed=x)
        # inference
        trace = pm.sample(draws=5000)

    # Show scatter of initial measurement and estimate of individual measurement error
    plt.figure()
    sigma = trace['σ']
    plt.scatter(x, np.mean(np.squeeze(sigma),axis=0))
    plt.xlabel('Initial measurment')
    plt.ylabel('Posterior estimate of measurement error')
    plt.show()

    # Show graphical model
    renderGraphicalModel(model)


# ------------------------------------------------------------
# 4.3 Repeated measurement of IQ
# ------------------------------------------------------------
"""
In this example, we consider how to estimate the IQ of a set of people, each of whom have done multiple IQ tests. The data are
the measures xij for the i = 1,...,n people and their j = 1,...,m repeated test scores.
We assume that the differences in repeated test scores are distributed as Gaussian error terms with zero mean and unknown precision.
The mean of the Gaussian of a person’s test scores corresponds to their latent true IQ. This will be different for each person. The
standard deviation of the Gaussians corresponds to the accuracy of the testing instruments in measuring the one underlying IQ value.
We assume this is the same for every person, since it is conceived as a property of the tests themselves.
"""
if sectionToRun == 3: # repeatedMeasurementIQ

    # IQ test data
    y = np.array([[90,95,100], [105,110,115], [150,155,160]])
    nSubjects = 3 # number of subjects
    nTests = 3 # number of tests per subject

    # Define model
    with pm.Model() as model:
        # prior
        μi = pm.Uniform('μi', 0, 300, shape=(nSubjects,1)) # μ estimates are different for each subject (given their different IQ levels)
        σ = pm.Uniform('σ', 0, 100) # σ (i.e. test error) is the same for all subjects (given that they take the same tests)
        # observed
        xij = pm.Normal('xij', mu=μi, sigma=σ, observed=y)
        # inference
        trace = pm.sample()

    # Show posterior estimates
    plt.figure(figsize=(6,12))
    μs = trace.get_values('μi')
    for subject in range(nSubjects):
        plt.subplot(nSubjects+1,1,subject+1)
        plotPosteriorDistribution(μs[:,subject,0], show=False)
        plt.title('Subject %s' % (subject+1))
    plt.subplot(nSubjects+1,1,nSubjects+1)
    plotPosteriorDistribution(trace.get_values('σ'), show=False)
    plt.title('Test STD'); plt.show()

    # Render model
    renderGraphicalModel(model)