import math
import scipy.stats as stats
import matplotlib.pylab as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
from tqdm import tqdm # for progresss bar

# Likelihood function
'''Note that this is a 2D function, with 3 Gaussian modes (located in a triangle pattern)'''
def getLikelihood(x,y):
    functionResult = np.zeros(shape=[len(x),len(y)])
    gausCenters = [[25,25],[70,50],[25,75]]
    gaussian = lambda x,mu,sig : stats.norm.pdf(x,loc=mu,scale=sig)
    for center in gausCenters:
        functionResult += np.outer(gaussian(x,mu=center[0],sig=10), gaussian(y,mu=center[1],sig=10))
    return functionResult

# Get full target distribution
xRange = np.arange(0, 100, 1)
yRange = np.arange(0, 100, 1)
X, Y = np.meshgrid(xRange, yRange)
targetDistribution = getLikelihood(xRange,yRange)
# plt.imshow(targetDistribution); plt.show()

# Proposal distribution
proposalDistribution = lambda : stats.norm.rvs(loc=0,scale=4)

# Metropolis-Hastings
steps = 15000
params = np.zeros(shape=(steps,2))
currentX, currentY = np.random.uniform(low=0,high=100,size=2)
print('Running Metropolis-Hastings MCMC algorithm...')
for step in tqdm(range(steps)):
    # Get current likelihood
    currentLikelihood = getLikelihood([currentX],[currentY])
    # Get next location
    nextX, nextY = currentX+proposalDistribution(), currentY+proposalDistribution()
    # Get proposed, next likelihood
    nextLikelihood = getLikelihood([nextX],[nextY])
    # Get likelihood ratio
    ratio = nextLikelihood/currentLikelihood
    # Generate uniform random number [0,1]
    r = np.random.uniform()
    # Change mu if r < ratio
    if ratio >= r:
        currentX, currentY = nextX, nextY
    params[step,:] = [currentX,currentY]

# Show target distribition and samples
fig,ax = plt.subplots(1,2,sharex=True)
ax[0].imshow(targetDistribution)
ax[0].set_title('Target distribution')
ax[1].imshow(targetDistribution)
#plt.scatter(params[:,1], params[:,0], s=1, color='k', alpha=.1)
ax[1].plot(params[:,1], params[:,0], color='k', linewidth=.2, alpha=.8)
ax[1].set_title('MCMC samples')
ax[0].axis('off'); ax[1].axis('off')
ax[1].set_xlim([np.max(xRange),np.min(xRange)]); ax[1].set_ylim([np.max(yRange),np.min(yRange)])
plt.tight_layout()
plt.show()

# # Show distribution of samples
# def getDistEst(values,xrange):
#     func = gaussian_kde(values)
#     return func(xrange)
# xDistEst = getDistEst(params[:10],xRange)
# yDistEst = getDistEst(params[:,0],yRange)
# combDistEst = np.outer(xDistEst,yDistEst)
# plt.imshow(combDistEst)

