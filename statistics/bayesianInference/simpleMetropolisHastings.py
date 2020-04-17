import math
import scipy.stats as stats
import matplotlib.pylab as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

# Create data
samples = 10000
data = stats.norm.rvs(size=samples,loc=1,scale=1)

# Likelihood function
def likelihood(data,mu):
    def normpdf(x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom
    likelihood = 0
    for i in data:
        # likelihood += stats.norm.pdf(i,loc=mu,scale=sig)
        likelihood += normpdf(i, mean=mu, sd=1)
    return likelihood

# Prior
def prior(x):
    return 1 # uniform prior over all value

# Proposal distribution
proposalDistribution = lambda : stats.norm.rvs(loc=0,scale=.05)

# Metropolis-Hastings
steps = 10000
params = np.zeros(shape=(steps,1))
currentMu = 4
for step in range(steps):
    print(step/steps)
    # Get current likelihood
    currentLikelihood = likelihood(data,currentMu)*prior(currentMu)
    # Get next location
    nextMu = currentMu + proposalDistribution()
    # Get proposed, next likelihood
    nextLikelihood = likelihood(data,nextMu)*prior(nextMu)
    # Get likelihood ratio
    ratio = nextLikelihood/currentLikelihood
    # Generate uniform random number [0,1]
    r = np.random.uniform()
    # Change mu if r < ratio
    if r < ratio:
        currentMu = nextMu
    params[step,:] = currentMu

# Show results
print(np.mean(params[-1000:,0]))
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].hist(data)
ax[1].hist(params[1000:,0])
plt.show()












# Metropolis-Hastings
steps = 200
previousLikelihood = -1
currentPositions = np.array([np.random.uniform(0,10),np.random.uniform(0,2)])
positionsRecords = np.zeros(shape=steps)
for step in range(steps):
    print(step)

    # Get new positions
    positionChanges = np.array([proposalDistribution() for i in range(2)])
    proposedPositions = currentPositions + positionChanges

    # Get likelihood
    curMu,curSig = proposedPositions
    curLikelihood = likelihood(data,curMu,curSig)

    # Compare current with previous likelihood
    if curLikelihood > previousLikelihood:
        currentPositions = proposedPositions
        previousLikelihood = curLikelihood
    else:
        if np.random.uniform() < (curLikelihood/previousLikelihood):
            currentPositions = proposedPositions
            previousLikelihood = curLikelihood

askldj


likelihood(data,mu=1,sig=1)

asdk






# # Define model function
# def parabola(x,a,h,k): # Parabola vertex form
#     return a*((x-h)**2)+k

# # Create datasets to fit
# x = np.arange(start=0,stop=4,step=.1)
# dataToFit = parabola(x,a=1,h=2,k=0)

# # Initial parameter guesses
# a = 10; h = 3; k = 1

# # Proposal distribution
# proposalDistribution = lambda : stats.norm.rvs(loc=0,scale=.01)

# # Metropolis-Hastings
# previousLikelihood = 0.01
# steps = 5000
# currentPositions = np.array([a,h,k])
# positionsRecords = np.zeros(shape=steps)
# for step in range(steps):
#     print(step)

#     # Get new positions
#     positionChanges = np.array([proposalDistribution() for i in range(3)])
#     proposedPositions = currentPositions + positionChanges

#     # Get likelihood
#     curA,curH,curK = proposedPositions
#     currentModelOutput = parabola(x,a=curA,h=curH,k=curK)
#     curLikelihood = stats.linregress(dataToFit,currentModelOutput).rvalue

#     # Compare current with previous likelihood
#     if curLikelihood > previousLikelihood:
#         currentPositions = proposedPositions
#         previousLikelihood = curLikelihood
#     else:
#         if np.random.uniform() < (curLikelihood/previousLikelihood):
#             currentPositions = proposedPositions
#             previousLikelihood = curLikelihood

#     positionsRecords[step] = previousLikelihood

# print(currentPositions)

# plt.plot(positionsRecords); plt.show()
# ajsd


# # Q(θ′/θ) = N(θ, σ)

# # Random walk
# currentPositions = 0
# steps = 1000; voltages = np.zeros(shape=steps)
# for step in range(steps):
#     currentPositions += proposalDistribution()
#     voltages[step] = currentPositions
# plt.plot(voltages); plt.show()




# x = np.arange(start=0,stop=4,step=.1)
# result = parabola(x,a=1,h=2,k=0)

# # priors
# priors = {
#     'a': lambda x: stats.norm.pdf(x,loc=2,scale=2),
#     'h': lambda x: stats.norm.pdf(x,loc=3,scale=2),
#     'k': lambda x: stats.norm.pdf(x,loc=1,scale=2)
# }

# x = np.arange(start=0,stop=4,step=.1)
# normDist = stats.norm.pdf(x,loc=2,scale=.5)
# twoD = np.outer(normDist,normDist)
# plt.imshow(twoD); plt.show()


# xS = np.vstack((x,x))
# cov = np.array([[1,0],[0,1]])
# stats.multivariate_normal.pdf(x=x)





# import numpy as np
# import scipy
# import scipy.stats
# import matplotlib as mpl   
# import matplotlib.pyplot as plt

# # -----------------------------------------------------
# # Step 1: Data generation
# # -----------------------------------------------------

# mod1=lambda t:np.random.normal(10,3,t)

# #Form a population of 30,000 individual, with average=10 and scale=3
# population = mod1(30000)
# #Assume we are only able to observe 1,000 of these individuals.
# observation = population[np.random.randint(0, 30000, 1000)]

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1)
# ax.hist( observation,bins=35 ,)
# ax.set_xlabel("Value")
# ax.set_ylabel("Frequency")
# ax.set_title("Figure 1: Distribution of 1000 observations sampled from a population of 30,000 with $\mu$=10, $\sigma$=3")
# mu_obs=observation.mean()
# plt.show()
#
#
# # -----------------------------------------------------
# # Step 5: Define the prior and the likelihood
# # -----------------------------------------------------

# #The tranistion model defines how to move from sigma_current to sigma_new
# transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))[0]]
# transition_model(np.arange(1,10,step=1))

# def prior(x):
#     #x[0] = mu, x[1]=sigma (new or current)
#     #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
#     #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
#     #It makes the new sigma infinitely unlikely.
#     if(x[1] <=0):
#         return 0
#     return 1

# #Computes the likelihood of the data given a sigma (new or current) according to equation (2)
# def manual_log_like_normal(x,data):
#     #x[0]=mu, x[1]=sigma (new or current)
#     #data = the observation
#     return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))

# #Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow.
# def log_lik_normal(x,data):
#     #x[0]=mu, x[1]=sigma (new or current)
#     #data = the observation
#     return np.sum(np.log(scipy.stats.norm(x[0],x[1]).pdf(data)))


# #Defines whether to accept or reject the new sample
# def acceptance(x, x_new):
#     if x_new>x:
#         return True
#     else:
#         accept=np.random.uniform(0,1)
#         # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
#         # less likely x_new are less likely to be accepted
#         return (accept < (np.exp(x_new-x)))


# def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):
#     # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
#     # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
#     # param_init: a starting sample
#     # iterations: number of accepted to generated
#     # data: the data that we wish to model
#     # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
#     x = param_init
#     accepted = []
#     rejected = []   
#     for i in range(iterations):
#         x_new =  transition_model(x)    
#         x_lik = likelihood_computer(x,data)
#         x_new_lik = likelihood_computer(x_new,data) 
#         if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
#             x = x_new
#             accepted.append(x_new)
#         else:
#             rejected.append(x_new)            
                
#     return np.array(accepted), np.array(rejected)


# # -----------------------------------------------------
# # Step 6: Run the algorithm with initial parameters and collect accepted and rejected samples
# # -----------------------------------------------------
# accepted, rejected = metropolis_hastings(manual_log_like_normal,prior,transition_model,[mu_obs,0.1], 50000,observation,acceptance)

