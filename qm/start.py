import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

N = 2000 

x = np.linspace(0, 1, N-2)
dx = 1.0/(N-1)

d = 2 / dx**2 * np.ones(N-2)
e = -1 / dx**2 * np.ones(N-3)

la, yy = eigh_tridiagonal(d, e) 
yy *= dx**2

ala = np.zeros(N-2)
for i in range(N-2):
    ala[i] = (i*np.pi)**2


plt.figure()
for i in range(10):
    plt.plot(x, yy[:,i])


plt.show()
