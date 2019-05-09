import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks


#################
# Settings
#################

# Choose lattice type
lattice_type = 4
lattice_names = [0, 0, 0, "Honeycomb", "Square", 0, "Triangular"]
print(lattice_names[lattice_type])


# Root of number of sites! Copy of this in "runall"-file, change both if changing either
if lattice_type == 6: # computed triangular with more Ns than the other cases..
    nsites = [100,  112,  128,  144,  162,  184,  206,  234,  264,  298,  336,  380,  430,  484,546,  616,  696,  784,  886, 1000]
else:
    nsites = [100, 130,  166,  216,  278,  360,  464,  600,  774, 1000] 


# Load data (created by convolution.f90)
data = []
basefilename = ("t%d.txt" % lattice_type)
for i in nsites:
    data.append(np.loadtxt("data/" + str(i) + "x" + str(i) +  basefilename, skiprows=1))

nfiles = len(data)
print(nfiles, "number of files")

nq = len(data[0][:,0]) # Number of probabilities

data = np.array(data) # Make the list into a numpy array


#################
# Calculations
#################

### Linear regression. Finds slope and correlation R^2
chi = np.array(nsites).astype(int)
logchi = np.log(chi)
minus_beta_del_nu = np.zeros(nq)
r = np.zeros(nq)
for i in range(nq):
    logpq = np.log(data[:,i,0])
    minus_beta_del_nu[i], _, r[i], _, _ = linregress(logchi, logpq) 
r2 = np.square(r)


### Find first peak of R^2
idx_r2max = find_peaks(r2, prominence=0.01)[0][0]


### Find critical exponents pc
pc = idx_r2max / (nq-1)
beta_del_nu = -minus_beta_del_nu[idx_r2max]
print("beta/nu", 0.1042, beta_del_nu)

idx_smax = np.argmax(data[:,:,1], axis=1) # Index of largest smax values 
smax = np.zeros(nfiles)
for i in range(nfiles):
    smax[i] = data[i,idx_smax[i], 1]
logsmax = np.log(smax)

gamma_del_nu = linregress(logchi, logsmax)[0]
print("gamma/nu", 1.7917, gamma_del_nu)

qmax = idx_smax / (nq-1)
x = np.log(pc-qmax)
nu = -linregress(x, logchi)[0] # Slope of x vs log(chi) gives nu

gamma =  gamma_del_nu * nu
beta = beta_del_nu * nu

# Print Pc and the critical exponents to console
print("\tTheory\tNumerical")
print("Pc\t??\t", pc)
print("Beta\t", 5/36, "\t", beta)
print("Gamma\t", 43/18, "\t", gamma)
print("Nu\t", 4/3, "\t", nu)


#################
# Plotting
#################
plot = True

if plot:

    # Choose xlimits for plotting based on lattice type
    if lattice_type == 3:
        lowlim = 0.4 # Show smaller region of the plot, 0.4 to 0.8
        highlim = 0.8

    elif lattice_type == 4:
        lowlim = 0.3
        highlim = 0.7

    elif lattice_type == 6:
        lowlim = 0.2
        highlim = 0.6

    # x values for plotting, should be renamed to q (probabilities)
    x = np.linspace(0, 1, nq)

    # Major component probability (p_inf)
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.xlabel("q")
    plt.ylabel(r"$p_{\infty}$")
    for i in range(nfiles):
        y = data[i,:,0]
        plt.plot(x, y, label="N=%s" % int(nsites[i]**2))
    plt.legend()
    plt.savefig("figures/p_infty%d.pdf" % lattice_type)

    # <s>
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.xlabel("q")
    plt.ylabel("<s>")
    for i in range(nfiles):
        y = data[i,:,1]
        plt.plot(x,y, label="N=%s" % int(nsites[i]**2))
    plt.legend()
    plt.savefig("figures/s_avg%d.pdf" % lattice_type)

    # chi
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.xlabel("q")
    plt.ylabel(r"$\chi$")
    for i in range(nfiles):
        y = data[i,:,2]
        plt.plot(x,y, label="N=%s" % int(nsites[i]**2))
    plt.legend()
    plt.savefig("figures/chi%d.pdf" % lattice_type)
    
    
plt.show()

