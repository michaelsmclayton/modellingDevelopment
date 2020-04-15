from brian2 import *
import matplotlib.pylab as plt
import numpy as np
import pyswarms as ps
import scipy.stats as stats

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
C = 281*pF; gL = 30*nS; EL = -70.6*mV; VT = -50.4*mV; deltaT = 2*mV; Vcut = VT+5*deltaT

# Create neurons
neurons = NeuronGroup(N=1, model=adexModel, threshold='u>Vcut', reset="u=Vr; w+=b", method='euler')
initialParameters = {'tau_w': 20*ms, 'a': 4*nS, 'b': 0.5*nA, 'Vr': VT+5*mV} # Bursting
# initialParameters = {'tau_w': 144*ms, 'a': 2*C/(144*ms), 'b': 0.0*nA, 'Vr': -70.6*mV} # Fast spiking
neurons.set_states(initialParameters)
trace = StateMonitor(neurons, 'u', record=True)
spikes = SpikeMonitor(neurons)
neurons.u = EL
neurons.w = 0
neurons.I = 2*nA
store()

# Run network
simulationDuration = 200*ms
def runNetwork(parameters):
    restore()
    neurons.set_states(parameters)
    run(simulationDuration)
    return trace.u[0]/mV, spikes.t_

# Get initial data
initialTrace, initialSpikes = runNetwork(initialParameters)

# -----------------------------------------------------
# Particle swarm optimisation
# -----------------------------------------------------

# Parameters
numberOfParticles = 50

# State priors (i.e. particle starting distributions)
variableParameters = { # Variable parameter priors
    'tau_w': (0,300),
    'a': (-12,4),
    'b': (0,120)} #
    # 'Vr': (-60,-40)}
priors = np.zeros(shape=(numberOfParticles,len(variableParameters)))
for i, param in enumerate(variableParameters):
    low,high = variableParameters[param]
    priors[:,i] = np.random.uniform(low=low,high=high, size=numberOfParticles)
    # mean,std,unit = variableParameters[param]
    # priors[:,i] = stats.norm.rvs(size=numberOfParticles, loc=mean, scale=std)

# Function to get parameters
def getParmaters(params):
    return {'tau_w': params[0]*ms, 'a': params[1]*nS}#, 'b': params[2]*nA, 'Vr': params[3]*mV}

# Create optimizer
'''c1: cognitive parameter, c2: social parameter, w: inertia parameter (hyperparameters)'''
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=numberOfParticles, dimensions=len(variableParameters), options=options, init_pos=priors)


# -----------------------------------------------------
# Fitness functions
# -----------------------------------------------------

# Get dot product
def getDotProduct(initialData, currentData):
    currentData_hat = currentData / np.linalg.norm(currentData)
    initialData_hat = initialData / np.linalg.norm(initialData)
    return -np.dot(currentData_hat,initialData_hat) # take negaive dot product

# Get R2
def getR2(initialData, currentData):
    linearFit = stats.linregress(initialData,currentData)
    return -linearFit.rvalue # take negative R^2 value

# Get fitness function
def fitnessFunction(x):
    fitnessValues = np.zeros(shape=len(x))
    for i, params in enumerate(x): # Loop over a values
        currentTrace, currentSpikes = runNetwork(getParmaters(params))
        # fitnessValues[i] = gammaFactor(currentSpikes, initialSpikes)
        fitnessValues[i] = getR2(initialTrace, currentTrace)
    return fitnessValues


# -----------------------------------------------------
# Run optimization
# -----------------------------------------------------

# Run optimizer
cost, pos = optimizer.optimize(fitnessFunction, 20)

# Plot results
plt.figure()
plt.plot(initialTrace)
resultTrace, resultSpikes = runNetwork(getParmaters(pos))
plt.plot(resultTrace, alpha=.5)
plt.show()



# # Gamma factor
# '''A benchmark test for a quantitative assessment of simple neuron models'''
# def gammaFactor(source, target, delta=.01):
#     def firing_rate(spikes):
#         return (len(spikes) - 1) / (spikes[-1] - spikes[0])
#     source = np.array(source)
#     target = np.array(target)
#     target_rate = firing_rate(target)
#     delta_diff = delta
#     source_length = len(source)
#     target_length = len(target)
#     if (source_length > 1):
#         bins = .5 * (source[1:] + source[:-1])
#         indices = np.digitize(target, bins)
#         diff = abs(target - source[indices])
#         matched_spikes = (diff <= delta_diff)
#         coincidences = sum(matched_spikes)
#     else:
#         indices = [amin(abs(source - target[i])) <= delta_diff for i in xrange(target_length)]
#         coincidences = sum(indices)
#     def get_gamma_factor(coincidence_count, model_length, target_length, target_rates, delta):
#         NCoincAvg = 2 * delta * target_length * target_rates
#         norm = .5 * (1 - 2 * delta * target_rates)
#         gamma = (coincidence_count - NCoincAvg) / (norm * (target_length + model_length))
#         return -gamma
#     return get_gamma_factor(coincidences, source_length, target_length, target_rate, delta)
# # Find coincidences
# sourceSpikes = initialSpikes
# modelSpikes = initialSpikes
# precision = .5
# sourceSpikeRanges = np.vstack((sourceSpikes-precision, sourceSpikes+precision)).T
# coincidenceCount = 0
# for modelSpike in modelSpikes:
#     for sourceSpikeRange in sourceSpikeRanges:
#         if (modelSpike>sourceSpikeRange[0] and modelSpike<sourceSpikeRange[1]):
#             coincidenceCount+=1

# # Expected coincidences
# f = len(sourceSpikes)/simulationDuration / Hz
# 2*f*precision*len(modelSpikes)