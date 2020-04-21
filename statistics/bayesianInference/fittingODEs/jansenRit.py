import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from scipy.integrate import odeint
import random
plt.style.use('seaborn-darkgrid')

# Parameters
C1,C2,C3,C4 = 135, 108, 33.75, 33.75
A,B,a,b = 6*mV, 20*mV, 100*Hz, 50*Hz
P, vmax, ε = 120*Hz, 5*Hz, 0.1
v0,r = 6*mV, 0.56*mV**-1
wij = .5

# Jansen-Rit
def jansenRit(inputs,times,params):
    def area(y,t,p,hz_in):
        '''Equations taken from [The role of node dynamics in shaping emergent functional connectivity patterns in the brain]'''
        hz_in *= Hz
        hz_out = y[-1]*Hz
        y0, y1, y2, y3, y4, y5 = y[0:-1]*mV # (y0, y3): pyramidal; (y1, y4): excitatory; (y2, y5): inhibitory
        I = 5000*(np.sin(t/20))*Hz
        C1,C2,C3,C4,P,A,B,a,b,ε,vmax,v0,r,wij = p
        # P += (20*random.random())*Hz # Add Gaussian noise to drive
        f = lambda v : vmax/(1.+np.exp(r*(v0-v))) # rate-to-potential
        dy0 = y3; dy1 = y4; dy2 = y5
        dy3 = ( A*a*f(y1-y2) - 2*a*y3/ms - (a**2)*y0)/Hz**2
        dy4 = ( A*a*(P + ε*wij*hz_in + C2*f(C1*y0)) - 2*a*y4/second - (a**2)*y1)/Hz**2
        dy5 = ( B*b*C4*f(C3*y0) - 2*b*y5/second - b**2*y2)/Hz**2
        dhz_out = f(y1-y2)-hz_out
        return dy0, dy1, dy2, dy3, dy4, dy5, dhz_out
    area1 = area(inputs[0:7],times,params[0],inputs[-1])
    area2 = area(inputs[7:],times,params[1],inputs[7])
    return np.hstack((area1,area2))

# Run simulation
y0 = np.zeros(shape=7) # np.array([ 1.77108696e-01,  4.01985725e+01,  3.35363309e+01,  2.10416795e-05, -4.76732276e-07,  6.98190921e-03, 0])
inputs = np.hstack((y0,y0))
times = np.arange(1,30000,.1)
params1 = [C1,C2,C3,C4,P,A,B,a,b,ε,vmax,v0,r,wij]
params2 = [C1,C2,C3,C4,P+60*Hz,A-1*mV,B+1*mV,a,b,ε,vmax,v0,r,-wij]
params = [params1,params2]
y = odeint(jansenRit, t=times, y0=inputs, args=tuple([params]), rtol=.1)

# Plot result
area1Trace = y[:,1]-y[:,2]
area2Trace = y[:,8]-y[:,9]
plt.plot(area1Trace)
plt.plot(area2Trace)
plt.show()


# # Code for single region
# def step(t, low, high):
#     if (t>low and t<high):
#         return 1
#     return 0

# # Jansen-Rit
# def jansenRit(y,t,params):
#     '''(y0, y3): pyramidal; (y1, y4): excitatory; (y2, y5): inhibitory'''
#     y0, y1, y2, y3, y4, y5 = y*mV
#     C1,C2,C3,C4,P,A,B,a,b,ε,vmax,v0,r = params
#     I = 1000*step(t,800,810)*Hz #1000*(np.sin(t/10))*Hz
#     f = lambda v : vmax/(1.+np.exp(r*(v0-v))) # rate-to-potential
#     dy0 = y3; dy1 = y4; dy2 = y5
#     dy3 = ( A*a*f(y1-y2) - 2*a*y3/ms - (a**2)*y0)/Hz**2
#     dy4 = ( A*a*(P + ε*(I) + C2*f(C1*y0)) - 2*a*y4/second - (a**2)*y1)/Hz**2
#     dy5 = ( B*b*C4*f(C3*y0) - 2*b*y5/second - b**2*y2)/Hz**2
#     return dy0, dy1, dy2, dy3, dy4, dy5

# # Parameters
# C1,C2,C3,C4 = 135, 108, 33.75, 33.75
# A,B,a,b = 6*mV, 20*mV, 100*Hz, 50*Hz
# P, vmax, ε = 120*Hz, 5*Hz, 0.1
# v0,r = 6*mV, 0.56*mV**-1

# # Run simulation
# times = np.arange(1,2000,.5)
# y0 = np.zeros(shape=6)
# params = [C1,C2,C3,C4,P,A,B,a,b,ε,vmax,v0,r]
# y = odeint(jansenRit, t=times, y0=y0, args=tuple([params]), rtol=1e-8)

# # Plot results
# fig,ax = plt.subplots(3,1)
# ax[0].plot(y[:,0], label='y0: pyramidal'); ax[0].legend()
# ax[1].plot(y[:,1], label='y1: excitatory'); ax[1].legend()
# ax[2].plot(y[:,2], label='y2: inhibitory'); ax[2].legend()
# plt.figure()
# plt.plot(y[:,1]-y[:,2], label='excitatory-inhibitory'); plt.legend()
# plt.show()