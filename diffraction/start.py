import numpy as np
from numpy.lib.scimath import sqrt as csqrt

from scipy.special import jv # Bessel function, 1st kind

from math import ceil

def abs2(z):
    return z.imag**2 + z.real**2


wavelength = 1
a = 3.5 * wavelength
b = 2*np.pi / a
zeta0 = 0.5 * wavelength
omega_c = 2*np.pi / wavelength

# G vector h-values
H = 4*ceil(omega_c)
print(H)
h1 = np.arange(-H, H+1)
h2 = np.arange(-H, H+1)
h = np.array(np.meshgrid(h1,h2)).T.reshape(-1,2) # All combinations of h1 and h2, last index changing fastest
hlength = 2*H+1

theta0 = 0
psi0 = 0
k = omega_c * np.sin(theta0) * np.array([np.cos(psi0), np.sin(psi0)])
alpha0_k = omega_c * csqrt(1 - np.sin(theta0))

# I_hat_right = (-1j)**h1 * jv(h1, gamma*zeta0/2) (-1j)**h2 * jv(h2, gamma*zeta0/2)
def calc_I_hat(z, h):
    return (-1j)**h[:,0] * jv(h[:,0], z) * (-1j)**h[:,1] * jv(h[:,1], z)

I_hat_right = calc_I_hat(alpha0_k*zeta0/2, h)

def calc_alpha0_K(k, h):
    G = np.ones(2) * b * h
    K = k + G
    return csqrt(omega_c**2 - np.sum(K**2, axis=-1)) 

shape = I_hat_right.shape[0]
I_hat_left = np.zeros((shape, shape), dtype=np.complex_)
alpha0_K = np.zeros(hlength**2, dtype=np.complex_)
for i, h_marked in enumerate(h):
    alpha0_Kmarked = calc_alpha0_K(k, h_marked)
    # print("alpha0_Kmarked", alpha0_Kmarked)
    alpha0_K[i] = alpha0_Kmarked # For later use
    h_diff = h - h_marked
    I_hat_left[i] = calc_I_hat( - alpha0_Kmarked * zeta0/2, h_diff)

r_K = np.linalg.solve(I_hat_left, -I_hat_right)
e_K = alpha0_K / alpha0_k * abs2(r_K)
# print("alpha0_K", alpha0_K)
# print("alpha0_k", alpha0_k)
# print("abs2(r_K)", abs2(r_K))
print(np.sum(e_K))
