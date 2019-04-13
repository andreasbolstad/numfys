"""
File structure
1. imports
2. constants and global variables
3. utility 
4. calculation 
5. plotting
6. tasks and main program (select what to do)
"""


# +-----------+
# |1. Imports |
# +-----------+

import time
from math import ceil

import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.special import jv # Bessel function, 1st kind

import matplotlib.pyplot as plt


# +---------------------------------+
# |2. Constants and global variables|
# +---------------------------------+

wavelength = 1.0
a = 3.5 * wavelength
b = 2*np.pi / a
zeta0 = 0.7 * wavelength
omega_c = 2*np.pi / wavelength

# G vector h-values
# H = 10
H = ceil(omega_c)
print(H)
h = np.arange(-H, H+1)
hlen = 2*H+1
h2len = hlen**2



# +----------+
# |3. Utility|
# +----------+

def abs2(z):
    """
    Function: Compute absolute squared of a complex number
    z: complex np.array or number
    """
    return z.imag**2 + z.real**2



# +---------------+
# |4. Calculations|
# +---------------+

def calc_I_hat(z, h1, h2):
    return (-1j)**(h1+h2) * jv(h1, z) * jv(h2, z)


def calc_dirichlet(theta0, psi0, zeta0):

    k1 = omega_c * np.sin(theta0) * np.cos(psi0)
    k2 = omega_c * np.sin(theta0) * np.sin(psi0)

    alpha0_k = omega_c * np.cos(theta0)

    gamma_right = alpha0_k*zeta0/2
    I_hat_right = calc_I_hat(gamma_right, h, h[:, np.newaxis])

    G_marked = h * b 
    K1_marked = k1 + G_marked
    K2_marked = k2 + G_marked

    K_marked_squared = K1_marked**2 + (K2_marked**2)[:, np.newaxis]
    alpha0_Kmarked = csqrt(omega_c**2 - K_marked_squared )
    
    h_marked = h[:, np.newaxis]
    h1_diff = h - h_marked 
    h2_diff = h - h_marked

    gamma_left =  -alpha0_Kmarked * zeta0/2
    I_hat_left = calc_I_hat(
            gamma_left, # 0 1
            h1_diff[np.newaxis, :, np.newaxis, :], # 0, 2
            h2_diff[:, np.newaxis, :, np.newaxis]) # 1, 3 

    lhs = I_hat_left.reshape(h2len, h2len) # dim 0 1 -> 0; 2 3 -> 1
    rhs = -I_hat_right.reshape(h2len) # dim 0 1 -> 0
    r_K = np.linalg.solve(lhs, rhs)

    e_K = alpha0_Kmarked.reshape(h2len) / alpha0_k * abs2(r_K) 

    return e_K



# +-----------+
# |5. Plotting|
# +-----------+



# +-------+
# |6. Main|
# +-------+

def task1():
    """
    No particular implementation, just testing
    """
    zeta0 = 0.7
    psi0 = 1 
    N = 100
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    R = np.zeros(N)
    for i, theta0 in enumerate(theta_list):
        e_K = calc_dirichlet(theta0, psi0, zeta0)
        kk_index = (h2len - 1) // 2
        R[i] = e_K[kk_index].real

    plt.figure()
    plt.semilogy(theta_grads, R)
    plt.show()
        

def task2():
    """
    How U varies as a function of zeta0
    """
    theta0 = 0
    psi0 = 0

    N = 1000
    zeta0_list = np.linspace(0, 1, N)
    U = np.zeros(N)
    for i, zeta0 in enumerate(zeta0_list): 
        if i % 10 == 0: print(i, "/", N)
        e_K = calc_dirichlet(theta0, psi0, zeta0)
        U[i] = np.sum(e_K.real)
    plt.figure()
    plt.semilogy(zeta0_list, 1-U)
    plt.show()


if __name__ == "__main__":

    selected_options = [2]

    options = {
            1: "task1",
            2: "task2",
            3: "task3",
            }

    selected = [options[i] for i in selected_options]
    
    if "task1" in selected:
        task1()

    if "task2" in selected:
        task2()
    
