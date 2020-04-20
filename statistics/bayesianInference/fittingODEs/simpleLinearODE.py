

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import theano
plt.style.use('seaborn-darkgrid')
'''https://docs.pymc.io/notebooks/ODE_API_introduction.html'''

# For reproducibility
np.random.seed(20394)

# Define ODE
def freefall(y, t, p):
    return 2.0*p[1] - p[0]*y[0]

# Get data to fit
times = np.arange(0,10,0.5)
gamma, g, y0, sigma = 0.4, 9.8, -2, 2
y = odeint(freefall, t=times, y0=y0, args=tuple([[gamma,g]]))
yobs = np.random.normal(y, sigma)

# Define inference model
ode_model = DifferentialEquation(func=freefall,times=times,n_states=1, n_theta=2,t0=0)
with pm.Model() as model:
    # priors
    sigma = pm.HalfCauchy('sigma',1)
    g = pm.HalfCauchy('g',1)
    gamma = pm.Lognormal('gamma',0,1)
    # observed
    ode_solution = ode_model(y0=[0], theta=[gamma, g]) 
    '''The ode_solution has a shape of (n_times, n_states)'''
    Y = pm.Normal('Y', mu=ode_solution, sd=sigma, observed=yobs)
    # inference
    trace = pm.sample(1000, tune=1000, cores=1)
    # prior = pm.sample_prior_predictive()
    # posterior_predictive = pm.sample_posterior_predictive(trace)
    # data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)

# az.plot_posterior(data)
# plt.show()

# Plot results
fig, ax = plt.subplots(dpi=120)
plt.plot(times,y, label='True ODE', color='k', alpha=0.5)
plt.plot(times,yobs, label='Observed speed', linestyle='dashed', marker='o', color='red')
np.random.seed(20394)
fitGamma, fitG, fitSigma = np.mean(trace['gamma']), np.mean(trace['g']), np.mean(trace['sigma'])
yFit = odeint(freefall, t=times, y0=y0, args=tuple([[fitGamma,fitG]]))
yFit = np.random.normal(yFit, fitSigma)
plt.plot(times, yFit, label='Fit speed', linestyle='dashed', marker='o', alpha=.5, color='blue')
plt.legend()
plt.xlabel('Time (Seconds)')
plt.ylabel(r'$y(t)$');
plt.show()


