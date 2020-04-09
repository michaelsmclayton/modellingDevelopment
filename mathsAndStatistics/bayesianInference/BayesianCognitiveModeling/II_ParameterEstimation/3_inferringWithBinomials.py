import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats.kde import gaussian_kde

# Global variables
sectionToRun = 4 # 1: inferringARate, 2: differenceBetweenRates, 3: inferringCommonRate, 4: priorAndPosterior

# Function to render graphical model
def renderGraphicalModel(model):
    pm.model_to_graphviz(model).render(filename='model', view=True, cleanup=True)

# Plot posterior distribution
def plotPosteriorDistribution(trace,x=np.linspace(0,1,100)):
    posteriorDist = gaussian_kde(trace)
    plt.hist(trace, bins=100, normed=1, alpha=.3)
    plt.plot(x, posteriorDist(x), 'r') # distribution function
    plt.show()

# ------------------------------------------------------------
# 3.1 Inferring a rate
# ------------------------------------------------------------

"""
Here, we are trying to estimate the true ability of a student (θ). This ability (which is a value between
0 and 1) is unobservable. What we do observe is the student's accuracy on a 10-question exam.

Our prior beliefs about the student's ability are defined by a uniform distribution across all ability
levels (i.e. a beta distribution [alpha, beta = 1]). In other words, we do not think that any given ability
level is more likely than any other ability level.

We model the incoming data as coming from a binomial distribution. This distribution gives up the probability
of answering a given number of questions (n) correctly given a certain ability (θ) level (p). From this,
we can calculate the likelihood of observing a given test performance given the different possible ability
levels.
"""
if sectionToRun == 1: # inferringARate

    # Define data
    k = 5 # Number of correct answers
    n = 10 # Total number of questions

    # Define model
    with pm.Model() as model:
        # prior
        θ = pm.Beta('θ', alpha=1, beta=1)
        # observed (i.e. likelihood of observed value assuming a binomial distribution with a rate of θ and n questions)
        x = pm.Binomial('x', n=n, p=θ, observed=k)
        # inference
        trace = pm.sample()

    # Render graphical model
    renderGraphicalModel(model)

    # Plot posterior distribution
    plotPosteriorDistribution(trace['θ'], x)


# ------------------------------------------------------------
# 3.2 Difference between two rates
# ------------------------------------------------------------
"""
Here we have two students taking the same test. Student 1 gets k1 questions right, while student 2 gets k2 questions right.
From this data, we estimate the abilities of each student (as in the previous example). However, from these two estimates
of ability, we also want to estimate the difference in abilities between these two students. For this, we calculate the difference
between the θ parameters for the two students (i.e. θ1 - θ2). Do perform this calculation, we use to pm.Deterministic()
function, where can we subtract the θ2 distribution from the θ1 distribution.
"""
if sectionToRun == 2: # differenceBetweenRates

    # Define data
    k1, k2 = 5, 7 # two different scores
    n1 = n2 = 10 # Numbers of questions

    with pm.Model() as model:
        # prior
        θ1 = pm.Beta('θ1', alpha=1, beta=1)
        θ2 = pm.Beta('θ2', alpha=1, beta=1)
        # observed (i.e. likelihoods)
        x1 = pm.Binomial('x1', n=n1, p=θ1, observed=k1)
        x2 = pm.Binomial('x2', n=n2, p=θ2, observed=k2)
        # differences (as deterministic; which allow you to do algebra with probability distributions)
        δ = pm.Deterministic('δ', θ1-θ2)
        # inference
        trace = pm.sample()

    # Render graphical model
    renderGraphicalModel(model)

    #  Plot posterior distribution for δ
    x = np.linspace(np.min(trace['δ']), np.max(trace['δ']), 100)
    plotPosteriorDistribution(trace['δ'], x)


# ------------------------------------------------------------
# 3.3 Inferring a common rate
# ------------------------------------------------------------
"""
Here, we again observe two binary processes (i.e. two students performing a test), producing k1 and k2 successes (i.e. correct
answers) out of n1 and n2 trials (questions), respectively. However, now assume the underlying rate (θ) for both is the same.
As such, we trying to estimate the value for θ which would most likely generate both student's scores. Note that, in the
graphical model for this, the x values are contained in a box (i.e. using plate notation). Plates are bounding rectangles that
enclose independent replications of a graphical structure within a whole model. In this case, the plate encloses the two observed
counts and numbers of trials. Because there is only one latent rate θ (i.e., the same probability drives both binary processes)
it is not iterated inside the plate.
"""
if sectionToRun == 3: # inferringCommonRate

    # Define data
    k = np.array([5,7]) # Number of correct answers
    n = np.array([10,10]) # Total number of questions

    # Define model
    with pm.Model() as model:
        # prior
        θ = pm.Beta('θ', alpha=1, beta=1) # note that there is just one prior this time
        # observed (i.e. likelihoods)
        x = pm.Binomial('x', n=n, p=θ, observed=k)
        # inference
        trace = pm.sample()

    # Render graphical model
    renderGraphicalModel(model)

    # Plot posterior distribution
    plotPosteriorDistribution(trace['θ'])


# ------------------------------------------------------------
# 3.4 Prior and posterior prediction
# ------------------------------------------------------------

if sectionToRun == 4: # priorAndPosterior

    k = 1
    n = 15
    # Uncomment for Trompetter Data
    # k = 24
    # n = 121

    # Posterior predictive sampling (i.e. prior only model, with no observation)
    with pm.Model() as model_prior:
        # Prior
        θ = pm.Beta('θ', alpha=1, beta=1) # prior on rate θ
        x = pm.Binomial('x', n=n, p=θ) # k
        trace_prior = pm.sample()

    # Visualise
    renderGraphicalModel(model_prior)
    plotPosteriorDistribution(trace_prior['θ']) # Plot posterior distribution
    '''Essentially the posterior, in the absence of data, draws directly from the prior'''

    # # with observation
    # with pm.Model() as model_obs:
    #     θ = pm.Beta('θ', alpha=1, beta=1)
    #     x = pm.Binomial('x', n=n, p=θ, observed=k)
    #     trace_obs = pm.sample()
        
    # # prediction (sample from trace)
    # ppc = pm.sample_ppc(trace_obs, samples=500, model=model_obs)