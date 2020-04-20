import pymc3 as pm
from scipy.integrate import odeint
import matplotlib.pylab as plt
import numpy as np

# Variational or MCMC inference
inferenceType = 'MCMC' # Variational / MCMC

# Create data
xRange = np.arange(-3,3,.1)
def customFunction(x,args):
    a,b,c,d = args
    return a*(x**3) + b*(x**2) + c*x + d
initialData = customFunction(xRange,[1,1,-1,5])
# plt.plot(xRange,initialData); plt.show()

print('Running model...')
startingPositions = {'a':1,'b':1,'c':1,'d':1}
with pm.Model() as model:
    # priors
    a = pm.Lognormal('a',mu=1,sigma=1)
    b = pm.Lognormal('b',mu=1,sigma=1)
    c = pm.Lognormal('c',mu=1,sigma=1)
    d = pm.Lognormal('d',mu=1,sigma=1)
    # observed
    y = pm.Normal('Y', mu=customFunction(xRange,[a,b,c,d]), sigma=1, observed=initialData)
    # inference
    if inferenceType=='MCMC':
        print('Performong MCMC...')
        trace = pm.sample(draws=10000,start=startingPositions)
    elif inferenceType=='Variational':
        print('Setting up variational inference...')
        mean_field = pm.fit(method='advi', start=startingPositions)
        trace = mean_field.sample(10000)
    
# Plot results
fig,ax = plt.subplots(1,5)
pm.plot_posterior(trace['a'], ax=ax[0])
pm.plot_posterior(trace['b'], ax=ax[1])
pm.plot_posterior(trace['c'], ax=ax[2])
pm.plot_posterior(trace['d'], ax=ax[3])
ax[-1].plot(xRange, initialData, label='Target data')
fit_a, fit_b, fit_c, fit_d = np.mean(trace['a']), np.mean(trace['b']), np.mean(trace['c']), np.mean(trace['d'])
ax[-1].plot(xRange, customFunction(xRange,[fit_a, fit_b, fit_c, fit_d]), label='Fit data')
plt.legend()
plt.show()
