import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import simps
import matplotlib.pyplot as plt

N = 2000 
NUM_EIGVALS = 100

dx = 1.0/(N-1)

# Computing eigvals and vecs without the boundaries to avoid singular matrix. 
d = 2 / dx**2 * np.ones(N-2) # Diagonal elements
e = -1 / dx**2 * np.ones(N-3) # Next to diagonal elements (symmetric)

yy = np.zeros((N,NUM_EIGVALS)) # Array of egeinvectors, 2nd axis specifies which eigenvalue is used
# numpy.eigh_tridiagonal computes eigvals and eigvecs using a symmetric tridiagonal matrix
la, yy[1:-1] = eigh_tridiagonal(d, e, select='i', select_range=(0,NUM_EIGVALS-1))
yy *= dx**2

nlist = np.linspace(1, NUM_EIGVALS, NUM_EIGVALS, dtype=int) # All n's of different eigenvalues
ala = (nlist*np.pi)**2 # Analytical lambdas (eigenvalues)

plt.figure()
plt.title(r'$\lambda_n(E_n) = \frac{E_n}{E_0},\ E_0 = \frac{\hbar^2}{2mL^2}$')
plt.xlabel(r'n')
plt.ylabel(r"$\lambda_n$")
plt.plot(nlist, ala, label=r"$\lambda^{analytical}$")
plt.plot(nlist, la, label=r"$\lambda^{numerical}$")
plt.legend()

psi = yy.T
for i in range(NUM_EIGVALS):
    psi[i] = psi[i] / np.sqrt(simps(np.square(psi[i]), dx=dx))

x = np.linspace(0, 1, N)
ayy = np.empty((N,NUM_EIGVALS), dtype=float) 
for n in range(1, NUM_EIGVALS):
    ayy[:,n-1] = np.sqrt(2) * np.sin(n*np.pi*x)

plt.figure()
for n in range(1):
    plt.plot(x, psi[n])
    plt.plot(x, ayy[:,n])

plt.show()
