import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import gridspec
import scipy.stats as stats
from scipy.stats.kde import gaussian_kde
from scipy import sparse
# see https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3

# 1:inferringARate, 2:differenceBetweenRates, 3:inferringCommonRate
# 4:priorAndPosterior, 5:usesOfPosteriorPrediction, 
sectionToRun = 5

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
    # pm.traceplot(trace, varnames=['θ']);

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
"""
We have so far talked about prior and posterior distributions. These can be thought of as beliefs about the probabilities of
different parameters (e.g. θ in the previous example). In addition, we can also calculate predictive distributions. Rather
than a distribution for model parameters, predictive distributions are distributions for the observations. These can be:
- Prior predictive distribution: what data to expect given only our prior belief
- Posterior predictive distribition: what data to expect given that our belief has been updated following observations
"""
if sectionToRun == 4: # priorAndPosterior
    k = 1; n = 15 # parameters

    # Prior predictive distribution (i.e. prior only model, with no observation)
    with pm.Model() as model_prior:
        θ = pm.Beta('θ', alpha=1, beta=1) # prior on rate θ
        x = pm.Binomial('x', n=n, p=θ) # k
        trace_prior = pm.sample()
    prior_x = trace_prior['x']

    # Posterior predictive distribition (i.e. prior model with observation)
    with pm.Model() as model_obs:
        θ = pm.Beta('θ', alpha=1, beta=1)
        x = pm.Binomial('x', n=n, p=θ, observed=k)
        trace_obs = pm.sample()
    pred_θ = trace_obs['θ']
    ppc = pm.sample_ppc(trace_obs, samples=500, model=model_obs) # prediction (sample from trace)
    predictx = ppc['x']

    # Visualise
    plt.figure(figsize=(12,8))
    plt.subplot(2, 1, 1)
    my_pdf = gaussian_kde(pred_θ)
    x = np.linspace(0, 1, 1000)
    plt.plot(x, my_pdf(x), 'r', label='Posterior') # distribution function
    plt.plot(x, stats.beta.pdf(x, 1, 1), 'b', label='Prior')
    plt.xlabel('θ (i.e. rate)'); plt.ylabel('Probability density'); plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(predictx, bins=len(np.unique(predictx)), alpha=.3, density=True, color='r', label='Posterior predictive')
    plt.hist(prior_x, bins=n+1, alpha=.3, density=True, color='b', label='Prior predictive')
    plt.xlabel('Success Count'); plt.ylabel('Predicted frequency (Mass)'); plt.legend()
    plt.show()


# ------------------------------------------------------------
# 3.5 Uses of posterior prediction
# ------------------------------------------------------------
"""
One important use of posterior predictive distributions is to examine the descriptive adequacy of a model. It can be
viewed as a set of predictions about what data the model expects to see, based on the posterior distribution over
parameters. If these predictions do not match the data already seen, the model is descriptively inadequate.
As an example to illustrate this idea of checking model adequacy, we return to the problem of inferring a common
rate underlying two binary processes. In the figure generated below, the left panel shows the posterior distribution
over the common rate θ for two binary processes, which gives density to values near 0.5. The right panel shows the
posterior predictive distribution of the model, with respect to the two success counts (i.e. student's scores). The
colour of each square is proportional to the predictive mass (i.e. frequency) given to each possible combination of
success count observations. The actual data observed in this example, with 0 and 10 successes for the two counts, are
shown by the dot. However, this figure shows that posterior predictive distributions do not line up with the actual data
"""

if sectionToRun == 5: # usesOfPosteriorPrediction

    # Parameters
    k1 = 0; k2 = 10 # Number of correct answers
    n1 = n2 = 10 # Total number of questions

    # Define model
    with pm.Model() as model:
        # prior
        θ = pm.Beta('θ', alpha=1, beta=1)
        # observed (i.e. likelihood)
        x1 = pm.Binomial('x1', n=n1, p=θ, observed=k1)
        x2 = pm.Binomial('x2', n=n2, p=θ, observed=k2)
        # inference
        trace = pm.sample()

    # Get posterior predictions
    ppc = pm.sample_ppc(trace, samples=500, model=model)
    predx1 = ppc['x1']; predx2 = ppc['x2']

    # Visualise
    fig = plt.figure(figsize=(12, 4)) 
    gs = gridspec.GridSpec(1,2, width_ratios=[2, 3])
    # subplot 1
    ax0 = plt.subplot(gs[0])
    my_pdf = gaussian_kde(trace['θ'])
    x = np.linspace(0.2, 1, 200)
    ax0.plot(x, my_pdf(x), 'r') # distribution function
    ax0.hist(trace['θ'], bins=100, alpha=.3, density=True)
    plt.xlabel('Rate')
    plt.ylabel('Posterior Density')
    # subplot 2
    ax1 = plt.subplot(gs[1])
    A = sparse.csc_matrix((np.ones(len(predx1)), (predx1,predx2)), shape=(n1+1,n2+1)).todense()
    '''2-D plot of posterior predictive probabilities for predx1 vs. predx2'''
    ax1.imshow(A, interpolation='none', alpha=.9, origin='lower')
    ax1.scatter(k1, k2, s=100, c=[1,0,0])
    plt.xlabel('Trial1'); plt.ylabel('Trial2')
    plt.tight_layout()
    plt.show()