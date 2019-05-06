import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt


k = np.arange(0, 201)
n = np.ones_like(k) * 200
res = comb(n, k)


plt.figure()
plt.semilogy(k, res)
plt.xlabel("# of A atoms")
plt.ylabel("# of possible Hamiltonians")
plt.show()


