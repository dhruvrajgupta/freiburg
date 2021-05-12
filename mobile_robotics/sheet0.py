import numpy as np
import matplotlib.pyplot as plt

def myfunc(x):
    return np.cos(x) * np.exp(x)

"""
x = np.linspace(-2*np.pi, 2*np.pi, num=100, endpoint=True)
y = myfunc(x)

plt.plot(x,y)
plt.grid()
plt.ylabel('cos(x) * exp(x)')
plt.xlabel('x')

plt.savefig("myfunc.png")
plt.show()
"""

np.random.seed(123)

nvector = np.random.normal(loc=5.0, scale=2.0, size=100000)

uvector = np.random.uniform(low=0, high=10, size=100000)

nmean = np.mean(nvector)
umean = np.mean(uvector)
ndev = np.std(nvector)
udev = np.std(uvector)

# Plot histogram
plt.figure()
_, bins, _ = plt.hist(x=uvector, bins=100, density=True)
plt.plot(bins, np.ones_like(bins)*0.1, linewidth=1, color='r')

_, bins, _ = plt.hist(x=nvector, bins=100, density=True)
plt.plot(bins, 1/(2 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - 5)**2 / (2 * 2**2) ), linewidth=1, color='g')

plt.show()