import numpy as np
import matplotlib.pylab as plt

# Make data to analyse
frequency = 25
def getSineWave(frequency):
    return np.cos(np.linspace(start=0,stop=frequency*(2*np.pi), num=1000))
x = getSineWave(frequency=50)
x += getSineWave(frequency=120) # Create mix of sines
x += np.random.randn(x.size) # Add noise

# --------------------------------------------
# Discrete Fourier Transform
# --------------------------------------------
'''https://youtu.be/PsEsMIPYJBg
also see https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/'''

# Create DFT matrix
N = x.size
W_N = np.exp(-2*np.pi * 1j / N)
I,J = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
dftMatrix = W_N**((I-1)*(J-1))
''' which returns:
    [ [   1         1            1         ...           1        ]
      [   1        W_N        W_N**2       ...      W_N**(N-1)    ]
      [   1      W_N**2       W_N**4       ...      W_N**2(N-1)   ]
      [  ...       ...          ...        ...          ...       ]
      [  ...       ...          ...        ...          ...       ]
      [   1    W_N**(N-1)  W_N**(2(N-1))   ...      W_N**(N-1**2) ]'''

# Plot DFT matrix (WARNING: it is pretty!)
# plt.imshow(np.real(dftMatrix)); plt.show()

# Perform DFP
dft = np.dot(x,dftMatrix)
fft = np.fft.fft(x) # also perform numpy fft (for comparison)

# Function to compute power spectral density
def psd(Y):
    '''https://en.wikipedia.org/wiki/Spectral_density#Energy_spectral_density'''
    return Y * np.conjugate(Y)

# # Plot power density spectra
# plt.figure()
# plt.plot(np.real(psd(fft)),label='FFT')
# plt.plot(np.real(psd(dft)),label='DFT')
# plt.xlim([0,N/2]); plt.legend()
# plt.show()

# --------------------------------------------
# Inverse Fourier Transform
# --------------------------------------------

# Get filtered fourier coefficients (i.e. level to zero all frequencies where power is less than threshold)
powerSpectralDensity = psd(fft)
threshold = 10**5
strongFreqIndices = np.where(powerSpectralDensity>threshold)
filteredFFT = np.zeros(shape=dft.shape)
filteredFFT[strongFreqIndices] = 1
filteredFFT = fft * filteredFFT

# Perform iFFT
filteredData = np.fft.ifft(filteredFFT)

# Plot results
plt.plot(x, label='Original data')
plt.plot(np.real(filteredData), label='Filtered data')
plt.show()