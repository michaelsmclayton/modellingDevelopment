from brian2 import *
import matplotlib.pylab as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats
from scipy.optimize import curve_fit

# -----------------------------------------------------
# Spiking model
# -----------------------------------------------------

# Define model equations (Adaptive exponential integrate-and-fire model)
adexModel = '''
    du/dt = ( -gL*(u-EL) + gL*deltaT*exp((u - VT)/deltaT) - w + I ) / C : volt
    dw/dt = ( a*(u-EL) - w ) / tau_w : amp
    I : amp
    tau_w : second
    a : siemens
    b : amp
    Vr : volt
'''

# Fixed parameters
C = 281*pF
gL = 30*nS
EL = -70.6*mV
VT = -50.4*mV
deltaT = 2*mV

# Create neurons
neurons = NeuronGroup(N=1, model=adexModel, threshold='u>20*mV', reset="u=Vr; w+=b", method='euler')
parameters = {'tau_w': 144*ms, 'a': 4*nS, 'b': 0.0805*nA, 'Vr': EL}
neurons.set_states(parameters)
trace = StateMonitor(neurons, 'u', record=True)
neurons.u = EL
neurons.w = 0
neurons.I = 1*nA
store()

# Run network
def runNetwork(currentA, iteration):
    restore(); # print('Running iteration %s...' % (iteration))
    neurons.set_states({'a': currentA})
    run(100*ms) # report='stdout')
    # plt.plot(trace.t/ms, trace.u[0]/mV)
    return trace.u[0]/mV

# Get initial data
initialData = runNetwork(currentA=parameters['a'], iteration='groundTruth')


# -----------------------------------------------------
# Bayesian inference
# -----------------------------------------------------

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def getLikelihoodDistribution(r2Values):
    popt, pcov = curve_fit(gaus, xdata=aValues, ydata=r2Values)
    return stats.norm.pdf(x=possibleAValues, loc=popt[1], scale=popt[2])

def getPosteriorDistribution(likelihoodDist, priorDist):
    posteriorDist = likelihoodDist * priorDist
    popt, pcov = curve_fit(gaus, xdata=possibleAValues, ydata=posteriorDist)
    return popt[1], popt[2]


# -----------------------------------------------------
# Model fitting
# -----------------------------------------------------

# Parameters
samples = 50
possibleAValues = np.arange(-10,20,step=.5)

# Initialise prior parameters
priorMu = 10; priorStd = 10

# Get likelihood data
def getLikelihoodData(aValues):
    r2Values = np.zeros(shape=aValues.shape)
    for i, currentA in enumerate(aValues): # Loop over a values
        currentData = runNetwork(currentA*nS, i)
        linearFit = stats.linregress(initialData,currentData)
        r2Values[i] = linearFit.rvalue
    return r2Values

# Loop over iterations
iterations = 5
for i in range(iterations):
    print('Running iterations %s/%s...' % (i+1,iterations))

    # Sample from prior distribution
    priorDist = stats.norm.pdf(x=possibleAValues, loc=priorMu, scale=priorStd)
    aValues = stats.norm.rvs(size=samples, loc=priorMu, scale=priorStd)

    # Get likelihood data
    r2Values = getLikelihoodData(aValues)

    # Get likelihood distribution
    likelihoodDist = getLikelihoodDistribution(r2Values)

    # Get posterior distribution
    priorMu, priorStd  = getPosteriorDistribution(likelihoodDist, priorDist)

    # Plot posterior distribution
    posteriorDist = stats.norm.pdf(x=possibleAValues, loc=priorMu, scale=priorStd)
    plt.plot(possibleAValues, posteriorDist)

plt.show()