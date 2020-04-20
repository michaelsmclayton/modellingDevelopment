

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import theano
plt.style.use('seaborn-darkgrid')
'''https://docs.pymc.io/notebooks/ODE_API_introduction.html'''
input('\nWARNING: This script could take a long time to run. Press any key to continue...')
print('\nRunning script...')

# For reproducibility
np.random.seed(20394)

# Define ODE
def SIR(y, t, p):
    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - p[1]*y[1]
    return [ds, di]

# Get data to fit
times = np.arange(0,5,0.25)
beta,gamma = 4,1.0
y = odeint(SIR, t=times, y0=[0.99, 0.01], args=((beta,gamma),), rtol=1e-8)
yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3])

# Define inference model
sir_model = DifferentialEquation(func=SIR,times=np.arange(0.25, 5, 0.25),n_states=2,n_theta=2,t0=0,)
with pm.Model() as model:
    # priors
    sigma = pm.HalfCauchy('sigma', 1, shape=2)
    R0 = pm.Bound(pm.Normal, lower=1)('R0', 2,3) # R0 is bounded below by 1 because we see an epidemic has occured
    lam = pm.Lognormal('lambda',pm.math.log(2),2)
    beta = pm.Deterministic('beta', lam*R0)
    # observed
    sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])
    Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs)
    # inference
    trace = pm.sample(500,tune=500, target_accept=0.9, cores=1)

# Get fit results
betaFit, gammaFit = np.mean(trace['beta']), np.mean(trace['lambda'])
yFit = odeint(SIR, t=times, y0=[0.99, 0.01], args=((betaFit,gammaFit),), rtol=1e-8)

# Plot results
fig, ax = plt.subplots(dpi=120)
plt.plot(times[1::],yobs, marker='o', linestyle='none')
plt.plot(times, y[:,0], color='C0', linestyle='dashed', alpha=0.5, label=f'$S(t)$')
plt.plot(times, yFit[:,0], color='C0', alpha=0.5, label=f'$SFit(t)$')
plt.plot(times, y[:,1], color ='C1', linestyle='dashed', alpha=0.5, label=f'$I(t)$')
plt.plot(times, yFit[:,1], color='C1', alpha=0.5, label=f'$IFit(t)$')
plt.legend()
plt.show()

