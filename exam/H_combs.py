import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt


k = np.arange(0, 201)
n = np.ones_like(k) * 200
res = comb(n, k) # Binomial coefficients (all possible combinations) 


plt.figure()
plt.semilogy(k, res)
plt.title("Number of possible Hamiltonians vs number of type A atoms, $N_H(N_A)$")
plt.xlabel(r"$N_{A}$")
plt.ylabel(r"$N_H$")
plt.show()


