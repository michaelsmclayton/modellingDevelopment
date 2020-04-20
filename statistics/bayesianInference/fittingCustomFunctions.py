import pymc3 as pm
from scipy.integrate import odeint
import matplotlib.pylab as plt
import numpy as np

# Create data
xRange = np.arange(1,50,1)
def customFunction(x,args):
    a,b,c = args
    return a*x**3 + b*x**2 + c*x
initialData = customFunction(xRange,[2,4,6])
#plt.plot(xRange,initialData); plt.show()

print('Running model...')
with pm.Model() as model:
    # priors
    a = pm.Lognormal('a',mu=1,sigma=1)
    b = pm.Lognormal('b',mu=1,sigma=1)
    c = pm.Lognormal('c',mu=1,sigma=1)
    # observed
    y = pm.Normal('Y', mu=customFunction(xRange,[a,b,c]), sigma=1, observed=initialData)
    # inference
    print('Setting up variational inference...')
    mean_field = pm.fit(method='advi', start={'a':1,'b':1,'c':1})
    # trace = pm.sample(draws=10000,start={'a':1,'b':1})

# Perform variational inference
trace = mean_field.sample(1000)

# Plot results
fig,ax = plt.subplots(1,4)
pm.plot_posterior(trace['a'], ax=ax[0])
pm.plot_posterior(trace['b'], ax=ax[1])
pm.plot_posterior(trace['c'], ax=ax[2])
ax[3].plot(initialData, label='Target data')
fit_a, fit_b, fit_c = np.mean(trace['a']), np.mean(trace['b']), np.mean(trace['c'])
ax[3].plot(customFunction(xRange,[fit_a, fit_b, fit_c]), label='Fit data')
plt.legend()
plt.show()
