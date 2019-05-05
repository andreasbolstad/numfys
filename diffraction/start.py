"""
File structure
1. imports
2. constants and global variables
3. utility 
4. calculation 
5. plotting and results (assignment tasks mostly)
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
from scipy.special import factorial
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib import ticker



# +---------------------------------+
# |2. Constants and global variables|
# +---------------------------------+

wavelength = 1.0
omega_c = 2*np.pi / wavelength

# G vector h-values
# H = 10
H = 14
# H = 2*ceil(omega_c)
print("2*omega / c =", 2*ceil(omega_c))
print("H =", H)
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

def calc_Ihat_dpc(a, b, gamma, zeta0, h1, h2):
    """Doubly periodic cosine"""
    z = gamma * zeta0 / 2
    return (-1j)**(h1+h2) * jv(h1, z) * jv(h2, z)


def calc_Ihat_tcone(a, b, gamma, zeta0, h1, h2, **kwargs):
    nfacs = 7 # Number of factors in sum, 3rd term in eq. 59
    pb = kwargs["pb"]
    pt = kwargs["pt"]
    G = np.sqrt(h1**2 + h2**2) * (2*np.pi / a)

    ##################
    # Pre gamma (calculations independent of gamma, and zeta0!)

    # Static storage / cache
    if 'dims' not in calc_Ihat_tcone.__dict__:
        calc_Ihat_tcone.dims = []
        calc_Ihat_tcone.cache = dict()
    
    dim = len(h1.shape)
    if dim not in calc_Ihat_tcone.dims:
        calc_Ihat_tcone.dims.append(dim)

        # 2
        z = G * pt
        pregamma_term2 = 2*np.pi * pt**2 / a**2 * np.divide(jv(1,z), z, out=np.zeros_like(z), where=z!=0)

        # 3
        pregamma_term3 = np.zeros((nfacs, *G.shape), dtype=float)
        quad_array = np.vectorize(lambda Gx: quad(lambda x: (pb-(pb-pt)*x) * jv(0, Gx*(pb-(pb-pt)*x)) * x**n , 0, 1)[0])
        for i, n in enumerate(np.arange(1, nfacs+1)):
            pregamma_term3[i] = quad_array(G) / factorial(n)
        pregamma_term3 *= 2 * np.pi * (pb-pt)/(a*a)
        
        calc_Ihat_tcone.cache[dim] = (pregamma_term2, pregamma_term3)
        
    else:
        pregamma_term2 = calc_Ihat_tcone.cache[dim][0]
        pregamma_term3 = calc_Ihat_tcone.cache[dim][1]

    ##################
    # Post gamma
    # 1
    term1 = np.where(G == 0, 1, 0)
   
    # 2
    postgamma_term2 = (np.exp(-1j*gamma*zeta0) - 1)
    term2 = pregamma_term2 * postgamma_term2

    # 3
    postgamma_term3 = np.zeros_like(pregamma_term3, dtype=np.complex_)
    for i, n in enumerate(np.arange(1, nfacs+1)):
        postgamma_term3[i] = (-1j*gamma*zeta0)**n
    term3 = np.sum(pregamma_term3 * postgamma_term3, axis=0)
    print(term3.dtype)

    return term1 + term2 + term3


def calc_Ihat_tcos(a, b, gamma, zeta0, h1, h2, **kwargs):
    nfacs = 7
    p0 = kwargs["p0"]
    G = np.sqrt(h1**2 + h2**2) * (2*np.pi / a)
    
    ##################
    # Pre gamma (calculations independent of gamma, and zeta0!)

    # Static storage / cache
    if 'dims' not in calc_Ihat_tcos.__dict__:
        calc_Ihat_tcos.dims = []
        calc_Ihat_tcos.cache = dict()
    
    dim = len(h1.shape)
    if dim not in calc_Ihat_tcos.dims:
        calc_Ihat_tcos.dims.append(dim)
        
        pregamma = np.zeros((nfacs, *G.shape), dtype=float)
        for i, n in enumerate(np.arange(1, nfacs+1)):
            quad_array = np.vectorize( lambda Gx: quad(lambda x: x*jv(0, Gx*x)*(zeta0*np.cos(0.5*np.pi*x/p0))**n, 0, p0)[0] ) 
            pregamma[i] = quad_array(G) * 2*np.pi / a**2 / factorial(n)
       
        calc_Ihat_tcos.cache[dim] = pregamma

    else:
        pregamma = calc_Ihat_tcos.cache[dim]

    postgamma = np.zeros_like(pregamma, dtype=np.complex_)
    for i, n in enumerate(np.arange(1, nfacs+1)):
        postgamma[i] = (-1j*gamma)**n

    return np.where(G == 0, 1, 0) + np.sum(pregamma * postgamma, axis=0)



def calc_rayleigh(a, b,theta0, phi0, zeta0, surface="dir", func="dpcos", **fargs):
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
    
    assert func in [ "dpcos", "tcone", "tcos" ] # Check for valid function type

    if func == "dpcos":
        calc_Ihat_func = calc_Ihat_dpc
    elif func == "tcone":
        calc_Ihat_func = calc_Ihat_tcone
    elif func == "tcos":
        calc_Ihat_func = calc_Ihat_tcos


    k1 = omega_c * np.sin(theta0) * np.cos(phi0)
    k2 = omega_c * np.sin(theta0) * np.sin(phi0)

    alpha0_k = omega_c * np.cos(theta0)

    gamma_right = alpha0_k#*zeta0/2
    I_hat_right = calc_Ihat_func(a, b, gamma_right, zeta0, h, h[:, np.newaxis], **fargs)


    G = h * b # only one axis, i.e. corresponds to G1 or G2 (both) 
    K1 = k1 + G
    K2 = k2 + G

    K_squared = (K1**2)[:, np.newaxis] + K2**2
    alpha0_Kmarked = csqrt(omega_c**2 - K_squared ) # "marked" to indicate that it iterates over the innermost axes
    
    # Note: Independent of angle!
    h_marked = h[:, np.newaxis]
    h1_diff = h - h_marked 
    h2_diff = h - h_marked

    gamma_left =  -alpha0_Kmarked #* zeta0 * 0.5
    I_hat_left = calc_Ihat_func( a, b, gamma_left, zeta0,
            h1_diff[:, np.newaxis, :, np.newaxis], # 1st and 3rd
            h2_diff[np.newaxis, :, np.newaxis, :], # 2nd and 4th
            **fargs)


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


def calc_anomalies(a, phi0):
    khat = np.array([np.cos(phi0), np.sin(phi0)])
    x1 = (h*khat[0] + (h*khat[1])[:, np.newaxis]) / a 
    # print(x1)
    x2 = (h/a)**2
    x3 = x1**2 - (x2 + x2[:, np.newaxis]) + 1

    # mask = np.where(x3 > 0, True, False)

    minuspart = -x1 - csqrt(x3)
    pluspart =  -x1 + csqrt(x3)
    # print(sin_theta0)
    anomaly_theta0 = []
    for item in minuspart.flatten():
        if item.imag == 0 and 0 < item.real < np.pi/2:
            anomaly_theta0.append(item)
    for item in pluspart.flatten():
        if item.imag == 0 and 0 < item.real < np.pi/2:
            anomaly_theta0.append(item)
    return np.arcsin(np.array(anomaly_theta0)) * 180 / np.pi


# +-----------------------+
# |5. Plotting and results|
# +-----------------------+

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



def toggleplot(fig, lines, leg):
    leg.get_frame().set_alpha(0.4)
    # we will set up a dict mapping legend line to orig line, and enable
    # picking on the legend line
    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)



def task1():
    """
    No particular implementation, miscellaneous testing
    """
    a = 3.5 * wavelength
    b = 2*np.pi / a
    zeta0 = 0.5 * wavelength
    phi0 = 0 #np.pi / 4 
    N = 100
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    R = np.zeros(N)
    for i, theta0 in enumerate(theta_list):
        e_K = calc_rayleigh(a,b, theta0, phi0, zeta0, surface="neu")
        kk_index = (h2len - 1) // 2
        R[i] = e_K[kk_index].real

    plt.figure()
    plt.semilogy(theta_grads, R)
        

def task2():
    """
    How U varies as a function of zeta0
    """
    theta0 = 0
    phi0 = 0

    N = 500
    zeta0_list = np.linspace(0, 1, N)
    U = np.zeros(N)

    fig, ax = plt.subplots()
    ax.set_title("Error in conservation of energy")
    ax.set_xlabel(r"$\zeta_0 / \lambda$")
    ax.set_ylabel("|1-U|")
    for a in [0.5 * wavelength, 3.5 * wavelength]:
    # for a in [3.5]:
        b = 2*np.pi / a
        for surface in ["dir", "neu"]:
        # for surface in ["neu"]:
            for i, zeta0 in enumerate(zeta0_list): 
                print(i, "/", N)
                e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface=surface)
                U[i] = np.sum(np.where(e_K.imag == 0, e_K.real, 0))
            result = np.abs(1-U)
            line, = ax.semilogy(zeta0_list, result, linestyle="", marker=".", label="a=%.1f, BC=%s" % (a, surface))
    leg = ax.legend(fancybox=True, shadow=True)
    fig.savefig("figures/energy_zeta.pdf")


def task3a():
    a = 3.5 * wavelength
    b = 2*np.pi / a
    zeta0_list = np.array([0.3, 0.5, 0.7]) * wavelength
    # phi0 = np.pi / 4 
    phi0 = 0
    N = 1000
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    t0s = calc_anomalies(a, phi0)
    
    R = np.zeros(N)
    fig, axes = plt.subplots(3, sharex=True, sharey=True)
    # presetup()
    for i, zeta0 in enumerate(zeta0_list):

        for j, theta0 in enumerate(theta_list):
            if j % 10 == 0: print(j, "/", N)
            e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface="neu")
            kk_index = (h2len - 1) // 2
            R[j] = e_K[kk_index].real

        line, = axes[i].semilogy(theta_grads, R, label=r"$\zeta_0$ = % .1f$\lambda$" % zeta0)
        axes[i].set_ylabel("Reflectivity")
        axes[i].legend(loc="center right")
        axes[i].autoscale(tight=True) # Remove margins between graph and axes
         
        for theta in t0s:
            axes[i].axvline(theta, linestyle='--', alpha=0.1, color='k', lw=0.8)
    axes[-1].set_xlabel(r"$\theta_0, deg$")
    fig.savefig("figures/reflectivity_phi00.pdf") 


def task3b():
    a = 3.5 * wavelength
    b = 2*np.pi / a
    zeta0 = 0.5 * wavelength
    phi0 = 0
    N = 1000
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    ekg = np.zeros((6, N))
    idx00 = (h2len - 1) // 2 # Center of hlist, where h1=h2=0
    # (0,0), (1,0), (-1,0), (0,pm1), (1,pm1), (-1,pm1)
    indices = [idx00, idx00+hlen, idx00-hlen, idx00+1, idx00+hlen+1, idx00-hlen-1]
    
    fig, axes = plt.subplots(6, figsize=(6,12), sharex=True)
    presetup()
    for i, theta0 in enumerate(theta_list):
        e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface="neu")
        for j, idx in enumerate(indices):
            ekg[j,i] = e_K[idx].real

    t0s = calc_anomalies(a, phi0)

    labels = ["h = {0, 0}", "h = {1, 0}", "h = {-1, 0}", "h = {0, ±1}", "h = {1, ±1}", "h = {-1, ±1}"]
    for i, idx in enumerate(indices):
        axes[i].plot(theta_grads, ekg[i], label=labels[i])
        axes[i].set_ylabel(r"$e(\mathbf{k_{||}}+\mathbf{G_{||}}(h)|\mathbf{k_{||}})$")
        for theta in t0s:
            axes[i].axvline(theta, linestyle='--', alpha=0.1, color='k', lw=0.8)
        axes[i].legend(loc='center left')
        postsetup(axes[i])
    axes[-1].set_xlabel(r"$\theta_0, deg$")
    fig.savefig("figures/efficiencies.pdf")


def task4():
    a = 3.5 * wavelength
    b = 2*np.pi / a
    zeta0 = 0.1 * wavelength
    phi0 = 0
    N = 500
    theta_grads = np.linspace(0, 90, N, endpoint=False) 
    theta_list = theta_grads * np.pi / 180 

    ekg = np.zeros((6, N))
    idx00 = (h2len - 1) // 2 # Center of hlist, where h1=h2=0
    # (0,0), (1,0), (-1,0), (0,pm1), (1,pm1), (-1,pm1)
    indices = [idx00, idx00+hlen, idx00-hlen, idx00+2*hlen, idx00-2*hlen, idx00+3*hlen]
    
    fig, axes = plt.subplots(6, figsize=(6,12), sharex=True)
    presetup()
    for i, theta0 in enumerate(theta_list):
        print("%d / %d" % (i, N) )
        # e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface="neu", func="dpcos")
        # e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface="neu", func="tcone", pb=a/4, pt=a/8)
        e_K = calc_rayleigh(a, b,theta0, phi0, zeta0, surface="neu", func="tcos", p0=a/4)
        for j, idx in enumerate(indices):
            ekg[j,i] = e_K[idx].real

    for i, idx in enumerate(indices):
        axes[i].plot(theta_grads, ekg[i])
        # postsetup(axes[i])


# +-------+
# |6. Main|
# +-------+

if __name__ == "__main__":

    selected_options = [3]

    options = {
        1: "task1",
        2: "task2",
        3: "task3a",
        4: "task3b",
        5: "task4",
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

    if "task4" in selected:
        task4()

    plt.show()


