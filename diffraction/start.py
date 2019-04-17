"""
File structure
1. imports
2. constants and global variables
3. utility 
4. calculation 
5. plotting (assignment tasks mostly)
6. main program (select what to do)
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
from matplotlib import rc, ticker



# +---------------------------------+
# |2. Constants and global variables|
# +---------------------------------+

wavelength = 1.0
a = 3.5 * wavelength
b = 2*np.pi / a
omega_c = 2*np.pi / wavelength

# G vector h-values
H = 10
# H = 2*ceil(omega_c)
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


def calc_periodic_rayleigh(theta0, psi0, zeta0, surface="dir"):
    """
    The following code can be a bit tricky to follow.
    Keep in mind that when a matrix is multiplied with a lower dimensional matrix with *, 
    the iteration goes over the innermost dimensions/axes, and repeats over the outer indices.

    E.g. in calculating I_hat_left the outermost axis is corresponds to h1, second axis 
    to h2 (unmarked coords.), and the 3rd and 4th axis to h1' and h2' (marked coords.) respectively.

    Numpy notation example:
    a and b are two 1D vectors
    c = a[:, np.newaxis] + b
    With this notation c is becomes a 2D array (outer sum).
    Exemplifying access order:
    c[1, 2] = a[1] + b[2] (a[:, np.newaxis] keeps the old indices of a on the axis marked with ":") 

    """

    assert surface in [ "dir", "neu"] # Check for valid surface type


    k1 = omega_c * np.sin(theta0) * np.cos(psi0)
    k2 = omega_c * np.sin(theta0) * np.sin(psi0)

    alpha0_k = omega_c * np.cos(theta0)

    gamma_right = alpha0_k*zeta0/2
    I_hat_right = calc_I_hat(gamma_right, h, h[:, np.newaxis])


    G = h * b # only one axis, i.e. corresponds to G1 or G2 (both) 
    K1 = k1 + G
    K2 = k2 + G

    K_squared = (K1**2)[:, np.newaxis] + K2**2
    alpha0_Kmarked = csqrt(omega_c**2 - K_squared ) # "marked" to indicate that it iterates over the innermost axes
    
    h_marked = h[:, np.newaxis]
    h1_diff = h - h_marked 
    h2_diff = h - h_marked

    gamma_left =  -alpha0_Kmarked * zeta0/2
    I_hat_left = calc_I_hat(
            gamma_left,
            h1_diff[:, np.newaxis, :, np.newaxis], # 1st and 3rd
            h2_diff[np.newaxis, :, np.newaxis, :]) # 2nd and 4th


    if surface == "neu":
        # K * K'
        prod1 = (K1 * K1[:, np.newaxis]) # K' (inner, 2nd) * K (outer 1st)
        prod2 = (K2 * K2[:, np.newaxis])
        prod_dot = prod1[:, np.newaxis, :, np.newaxis] + prod2[np.newaxis, :, np.newaxis, :]
        M = (omega_c**2 - prod_dot) / alpha0_Kmarked 

        N = -(omega_c**2 - ((k1 * K1)[:, np.newaxis] + k2 * K2)) / alpha0_k 

    elif surface == "dir":
        M = 1
        N = 1


    lhs = (I_hat_left * M).reshape(h2len, h2len) # Collapse to 2 axes. Row unmarked, col marked (K, K')
    rhs = (-I_hat_right * N).reshape(h2len) # One axis, corresponding to unmarked K
    r_K = np.linalg.solve(lhs, rhs)

    e_K = alpha0_Kmarked.reshape(h2len) / alpha0_k * abs2(r_K) 

    return e_K



# +-----------+
# |5. Plotting|
# +-----------+

# Plotting settings
def presetup():
    plt.subplots_adjust(wspace=0, hspace=0)

#Plotting settings
def postsetup(ax):
    ax.autoscale(tight=True)
    ax.tick_params(direction='in', which='both')
    start, end = ax.get_ylim()
    sep = float('%.2f' % (end/4))
    ax.yaxis.set_ticks(np.arange(0, end-sep*0.5, sep))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))


def task1():
    """
    No particular implementation, miscellaneous testing
    """
    zeta0 = 0.5 * wavelength
    psi0 = 0 #np.pi / 4 
    N = 100
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    R = np.zeros(N)
    for i, theta0 in enumerate(theta_list):
        e_K = calc_periodic_rayleigh(theta0, psi0, zeta0, surface="neu")
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

    N = 100
    zeta0_list = np.linspace(0, 1, N)
    U = np.zeros(N)
    for i, zeta0 in enumerate(zeta0_list): 
        if i % 10 == 0: print(i, "/", N)
        e_K = calc_periodic_rayleigh(theta0, psi0, zeta0, surface="neu")
        U[i] = np.sum(e_K.real)
    plt.figure()
    plt.semilogy(zeta0_list, 1-U)
    plt.show()


def task3a():
    zeta0_list = np.array([0.3, 0.5, 0.7]) * wavelength
    psi0 = np.pi / 4 
    N = 100
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    R = np.zeros(N)
    
    fig, axes = plt.subplots(3, sharex=True, sharey=True)
    presetup()
    for i, zeta0 in enumerate(zeta0_list):

        for j, theta0 in enumerate(theta_list):
            e_K = calc_periodic_rayleigh(theta0, psi0, zeta0, surface="neu")
            kk_index = (h2len - 1) // 2
            R[j] = e_K[kk_index].real

        axes[i].semilogy(theta_grads, R, label=r"$\zeta_0$ = % .1f" % zeta0)
        axes[i].legend()
        axes[i].autoscale(tight=True)

    plt.show()
    

def task3b():
    zeta0 = 0.5 * wavelength
    psi0 = 0
    N = 100
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    ekg = np.zeros((6, N))
    idx00 = (h2len - 1) // 2
    # (0,0), (1,0), (-1,0), (0,pm1), (1,pm1), (-1,pm1)
    indices = [idx00, idx00+hlen, idx00-hlen, idx00+1, idx00+hlen+1, idx00-hlen-1]
    
    fig, axes = plt.subplots(6, figsize=(6,12), sharex=True)
    presetup()
    for i, theta0 in enumerate(theta_list):
        e_K = calc_periodic_rayleigh(theta0, psi0, zeta0, surface="neu")
        for j, idx in enumerate(indices):
            ekg[j,i] = e_K[idx].real

    for i, idx in enumerate(indices):
        axes[i].plot(theta_grads, ekg[i])
        postsetup(axes[i])

    plt.show()



def task4():
    pass



# +-------+
# |6. Main|
# +-------+

if __name__ == "__main__":

    selected_options = [4]

    options = {
        1: "task1",
        2: "task2",
        3: "task3a",
        4: "task3b",
    }

    selected = [options[i] for i in selected_options]
    
    if "task1" in selected:
        task1()

    if "task2" in selected:
        task2()
    
    if "task3a" in selected:
        task3a()

    if "task3b" in selected:
        task3b()

