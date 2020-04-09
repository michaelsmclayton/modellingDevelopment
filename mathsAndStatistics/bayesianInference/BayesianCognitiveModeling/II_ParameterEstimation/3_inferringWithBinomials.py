import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats.kde import gaussian_kde

# Global variables
sectionToRun = 3 # 1: inferringARate, 2: differenceBetweenRates, 3: inferringCommonRate

# Function to render graphical model
def renderGraphicalModel(model):
    pm.model_to_graphviz(model).render(filename='model', view=True, cleanup=True)

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
    posteriorDist = gaussian_kde(trace['θ'])
    x = np.linspace(0,1,100)
    plt.hist(trace['θ'], bins=100, normed=1, alpha=.3)
    plt.plot(x, posteriorDist(x), 'r') # distribution function
    plt.show()


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
        delta = pm.Deterministic('delta', θ1-θ2)
        
        # inference
        trace = pm.sample()

    # Render graphical model
    renderGraphicalModel(model)

    #  Plot posterior distribution for delta
    x = np.linspace(np.min(trace['delta']), np.max(trace['delta']), 100)
    deltaPosterior = gaussian_kde(trace['delta'])(x)
    plt.hist(trace['delta'], bins=100, normed=1, alpha=.3)
    plt.plot(x, deltaPosterior, 'r') # distribution function
    plt.show()


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
    with pm.Model() as model3:

        # prior
        θ = pm.Beta('θ', alpha=1, beta=1) # note that there is just one prior this time

        # observed (i.e. likelihoods)
        x = pm.Binomial('x', n=n, p=θ, observed=k)

        # inference
        trace = pm.sample()

    # Render graphical model
    renderGraphicalModel(model3)

    # Plot posterior distribution
    posteriorDist = gaussian_kde(trace['θ'])
    x = np.linspace(0,1,100)
    plt.hist(trace['θ'], bins=100, normed=1, alpha=.3)
    plt.plot(x, posteriorDist(x), 'r') # distribution function
    plt.show()