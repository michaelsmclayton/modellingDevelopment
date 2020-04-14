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
''' 1: pearsonCorrelation; 2: pearsonCorrelationWithUncertainty; 3: kappaCoefficientOfAgreement;
    4: changeDetectionInTimeSeries'''
sectionToRun = 4

# Function to create linearly correlated dataset
def createLinearData(α, σ):
    x = np.random.uniform(low=0, high=10, size=11)
    y = α * x + (σ*np.random.rand(len(x)))
    return np.vstack((x,y)).T

# Function to plot posterior density over r (i.e. correlation coefficient)
def plotPosteriorOverR(y,trace,show=False):
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(y[:,0],y[:,1])
    plt.subplot(2,1,2)
    plotPosteriorDistribution(trace.get_values('r'), show=False)
    rValue = stats.pearsonr(y[:,0],y[:,1])[0]
    plt.plot((rValue,rValue), plt.gca().get_ylim()) # Plot calculated Pearson correlation coefficient
    if show==True: plt.show()

# Return correlated data from book
def getBookData():
    return np.array([.8, 102, 1,98, .5,100, 0.9,105, .7,103, 0.4,110, 1.2,99, 1.4,87, 0.6,113, 1.1,89, 1.3,93]).reshape((11, 2))

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
    y = createLinearData(α=-1, σ=3)
    # y = getBookData() # Data from book

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
    plotPosteriorOverR(y,trace)

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
In this example, we extend the previous code to take account of possible error in the measurment of our observations.
For example, if the two dimensions correspond to reaction times (RT) and IQ scores, we might expect that the precision
of RT measurements is greater (as it is a physical quantity) than IQ (which is a psychological concept). Therefore, we
may want to add this information into our model. In other words, the uncertainty in measurement should be incorporated
in an assessment of the correlation between the variables.

A simple approach for including this uncertainty is adopted in the code below. Here, the observed data still take the
form xi = (xi1, xi2) for the ith person’s response time and IQ measure. However, these observations are now sampled from
a Gaussian distribution, centered on the unobserved true response time and IQ of that person, denoted by yi = (yi1, yi2).
These true values are then modeled as x was in the previous model, and drawn from a multivariate Gaussian distribution.
The precision of the measurements is captured by λe = (λe1,λe2) of the Gaussian draws for the observed data.
"""
if sectionToRun == 2:

    # The datasets:
    y = getBookData()
    λe = np.array([.03, 1]) # Uncertainty in measurement precision (i.e. RTs vs IQ; RTs are thought to have less error than IQ)

    # Define the model
    with pm.Model() as model:
        # prior
        r =  pm.Uniform('r', lower=-1, upper=1) # uniform prior over correlation coefficient values
        μ = pm.Normal('μ', mu=0, tau=.001, shape=2) # mean of two variables is set to 0, with low precision
        σ1, σ2 = pm.Gamma('σ1', alpha=.001, beta=.001), pm.Gamma('σ2', alpha=.001, beta=.001) # uninformative priors for within-variable standard deviations
        cov = pm.Deterministic('cov', tensor.stacklists([[σ1**2, r*σ1*σ2], [r*σ1*σ2, σ2**2]])) # create covariance matrix (see section above for further details)
        yi = pm.MvNormal('yi', mu=μ, cov=cov, shape=np.shape(y)) # unobserved, 'true' RT and IQ of each person
        # observed
        xi = pm.Normal('xi', mu=yi, sd=λe, observed=y) # likelihood estimate based on estimated, true RT/IQ, as well as known measurement errors
        # inference
        trace = pm.sample()

    # Plot results
    plotPosteriorOverR(y,trace,show=True)

    # Plot graphical model
    renderGraphicalModel(model)


# ------------------------------------------------------------
# 5.3 The kappa coefficient of agreement
# ------------------------------------------------------------

# What is the kappa coefficient?
"""
It is often important in statistics to determine the agreement between two different decision making methods (i.e.
inter-rater reliability). When the decisions that are made are categorical (e.g. 1 vs. 0), one can use the kappa
coefficient of agreement. In this process, one of the decision-making methods is viwed as giving objectively true
decisions to which the other method aspires to match. A real-world example of this kind of problem might come when
a cheap, experimental method for medical diagnosis to some expensive, gold-standard method.

To calculate the kappa coefficient, when both decision-making methods make n independent assessments, the data y take
the form of four counts:
 - a observations (where both methods decide 1),
 - b observations (where the objective method decides 1 but the other method decides 0),
 - c observations (where the objective method decides 0 but the surrogate method decides 1) and,
 - d observations (where both methods decide 0)
combining these counts together yields all possible observations (n = a + b + c + d)

Cohen’s (1960) kappa statistic is estimated by comparing:
- the level of observed agreement
        po = (a+d) / n   (i.e. the proportion of observations where both methods agree)
- the agreement that would be expected by chance alone
        pe = (a+b)(a+c)+(b+d)(c+d) / n**2 (i.e. the overall probability for the first method to decide 1 (a+b), times the overall
                                            probability for the second method to decide 1 (a+c), and added to this the overall
                                            probability for the second method to decide 0 (b+d),  times the overall probability
                                            for the first method to decide 0 (c+d))
Specifically, kappa (κ) is calculated as follows:
    κ = po − pe / 1−pe

Kappa lies on a scale of −1 to +1, with values below 0.4 often interpreted as “poor” agreement beyond chance, values between
0.4 and 0.75 interpreted as “fair to good” agreement beyond chance, and values above 0.75 interpreted as “excellent” agreement
beyond chance (Landis & Koch, 1977). The key insight of kappa as a measure of agreement is its correction for chance agreement.
"""

# Description of the model below
"""
A Bayesian version of kappa is coded below. The key latent variables are α, β, and γ:
- α is the rate at which the gold standard method decides 1. This means (1−α) is the rate at which the gold standard method decides 0.
- β is the rate at which the surrogate method decides 1 when the gold standard also decides 1.
- γ is the rate at which the surrogate method decides 0 when the gold standard decides 0.
(One can interpret β and γ as the rates of agreement of the surrogate method with the gold standard, for the 1 and 0 decisions, respectively)

Using the rates α, β, and γ, it is possible to calculate the probabilities for a,b,c, and d (defined in the notes above).
- πa = αβ (i.e. the probability that both methods will decide 1)
- πb = α(1−β) (i.e. the probability that the gold standard will decide 1, but the surrogate will decide 0)
- πc = (1−α)(1−γ) (i.e. the probability that the gold standard will decide 0 but the surrogate will decide 1)
- πd = (1−α)γ  (i.e. the probability that both methods will decide 0)

These probabilities, in turn, describe how the observed data, y, made up of the counts a, b, c, and d, are generated. They come from
a multinomial distribution with n trials, where on each trial there is a πa probability of generating an 'a' count, πb probability for
a 'b' count, and so on.

So, observing the data y allows inferences to be made about the key rates α, β, and γ. In turn, these rates help to calculate the
following higher-order parameters:
- ξ measures the rate of agreement (ξ = αβ + (1−α)γ)
- ψ measures the rate of agreement that would occur by chance (ψ = (πa+πb)(πa+πc) + (πb+πd)(πc+πd))

From these higher-order parameters, κ can be calculated, which is the chance-corrected measure of agreement on the −1 to +1 scale (given
by κ = (ξ − ψ) / (1 − ψ))
"""
if sectionToRun == 3:

    # Choose data
    def getData(dataType):
        if dataType=='influenza':
            return np.array([14, 4, 5, 210])
        elif dataType=='hearingLoss':
            return np.array([20, 7, 103, 417])
        elif dataType=='rareDisease':
            return np.array([0, 0, 13, 157])
    data = getData('influenza')

    # Define model
    with pm.Model() as model:
        # prior
        α = pm.Beta('α', alpha=1, beta=1)
        β = pm.Beta('β', alpha=1, beta=1)
        γ = pm.Beta('γ', alpha=1, beta=1)
        # derived measures
        πa,πb,πc,πd = α*β, α*(1-β), (1-α)*(1-γ), (1-α)*γ # a,b,c,d
        ξ = pm.Deterministic('ξ', (α*β+(1-α)*γ)) # rate of agreement
        ψ = pm.Deterministic('ψ', (πa+πb)*(πa+πc)+(πb+πd)*(πc+πd)) # rate of agreement that would occur by chance
        κ = pm.Deterministic('κ', (ξ-ψ)/(1-ψ)) # chance-corrected rate of agreement
        # observed
        y = pm.Multinomial('y', n=data.sum(), p=[πa, πb, πc, πd], observed=data)
        # inference
        trace=pm.sample()

    # Plot α, β, γ
    plt.figure()
    for i,string in enumerate(['α','β','γ']):
        plt.subplot(3,1,i+1)
        plotPosteriorDistribution(trace.get_values(string), show=False)
        plt.title(string)
    plt.show()
    # α_calc = (data[0]+data[1])/np.sum(data)
    # β_calc = data[0]/np.sum(data)
    # γ_calc = data[3]/np.sum(data)

    # Plot (kappa) estimate
    def getKappEstimate(data):
        n = data.sum()
        p0 = (data[0]+data[3])/n
        pe = (((data[0]+data[1]) * (data[0]+data[2])) + ((data[1]+data[3]) * (data[2]+data[3]))) / n**2
        return (p0-pe) / (1-pe) # Cohen's point estimate
    plt.figure()
    kappa_Cohen = getKappEstimate(data)
    plotPosteriorDistribution(trace.get_values('κ'), show=False)
    plt.plot([kappa_Cohen,kappa_Cohen], plt.gca().get_ylim())
    plt.show()

    # Plot graphical model
    renderGraphicalModel(model)

# ------------------------------------------------------------
# 5.4 Change detection in time series data
# ------------------------------------------------------------
"""
An interesting use of Bayesian modelling is to detect a change occurs in noisy time series. In the code below, we generate time-series
data in which, while the standard deviation remains constant throughout, the mean suddenly changes at some point. The goal of the
analysis is to calculate from the data the posterior probability over when this change in mean activity occured.

In order to model this problem, we assume (correctly as we made the data!) that the data come from a Gaussian distribution that always has
the same variance, but changes its mean at one specific point in time. The observed data are the counts ci at time ti for the ith sample.
The unobserved variable τ is the time at which the change happens, which controls whether the counts have mean μ1 or μ2. A uniform prior
over the full range of possible times is assumed for the change point, and generic weakly informative priors are given to the means (μ) and
the precision (λ).

Note the use of the pm.math.switch() function here, which returns μ1 when the sample index is less than τ (i.e. the estimated time of change),
and returns μ1 when the sample index is greater than τ.
"""
if sectionToRun == 4:

    # Load data
    def getTimeShiftData(μ1, μ2, timepoints):
        changeTime = int(np.random.uniform(low=int(.3*timepoints),high=int(.7*timepoints)))
        data_preChange = μ1 + np.random.randn(changeTime)
        data_postChange = μ2 + np.random.randn(timepoints-changeTime)
        return np.hstack((data_preChange, data_postChange))
    data = getTimeShiftData(μ1=3, μ2=1, timepoints=500)
    n = np.size(data)
    sample = np.arange(0, n)

    # Define model
    with pm.Model() as model:
        # priors
        μ = pm.Normal('μ', mu=0, tau=.001, shape=2) # note shape =2, thus this equals μ1 and μ2
        λ = pm.Gamma('λ', alpha=.001, beta=.001)
        τ = pm.DiscreteUniform('τ', lower=0, upper=n) # discrete, uniform prior over all possible change times
        μvect = pm.math.switch(sample<=τ, μ[0], μ[1]) # mu[0] for samples less than tau, mu10] for samples greater than tau 
        # observed
        ci = pm.Normal('ci', mu=μvect, tau=λ, observed=data)
        # inference
        trace = pm.sample()

    # Plot results
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.subplot(2,1,2)
    plotPosteriorDistribution(trace.get_values('τ'), x=np.linspace(0,n,100), show=False)
    plt.ylim([0,1])
    plt.show()