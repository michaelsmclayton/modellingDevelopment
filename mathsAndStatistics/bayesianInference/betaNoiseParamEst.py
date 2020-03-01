import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
plt.style.use('seaborn-darkgrid')

"""
In this script, a random Beta distribution is chosen (i.e. with random
values of alpha and beta) and samples taken from that distribution. With
uniform prior probabilities for alpha and beta values, a Bayesian model
is used to estimate these Beta distribution parameters based on the
observed, generated samples.
"""

# -------------------------
# Generate data
# -------------------------
print('\nGenerating data...')
nOfSamples = 1000

# True parameter values (for beta distribution)
alpha_true = np.random.uniform(0,5)
beta_true = np.random.uniform(0,5)

# Generate noise data
data = np.random.beta(alpha_true, beta_true, size=nOfSamples)

# -------------------------
# Model specification
# -------------------------
print('Specifying model ...')
basicModel = pm.Model()
with basicModel:
    # Priors
    alpha = pm.Uniform('alpha', 0, 6)
    beta = pm.Uniform('beta', 0, 6)
    # Likelihood
    likelihood = pm.Beta('likelihood', alpha, beta, observed=data)

# ---------------------------------
# Model fitting
# ---------------------------------
print('Fitting model ...')
map_estimate = pm.find_MAP(model=basicModel, progressbar=False)
alpha_est = map_estimate['alpha'] # Get estimates
beta_est = map_estimate['beta']
print('\nRESULTS:')
print('True alpha: %s; Estimated alpha: %s' % (alpha_true, alpha_est))
print('True beta: %s; Estimated beta: %s' % (beta_true, beta_est))

# Show raw data and estimated fit
plt.figure()
rangeToFit = np.arange(0,1,.01)
elem1 = plt.hist(data, density=True)
fit_true = stats.beta.pdf(rangeToFit, alpha_true, beta_true)
fit_est = stats.beta.pdf(rangeToFit, alpha_est, beta_est)
elem2 = plt.plot(rangeToFit, fit_true, color='k')
elem3 = plt.plot(rangeToFit, fit_est, color='b')
plt.legend({'Observed data', 'Estimated PDF', 'True PDF'})
plt.show()