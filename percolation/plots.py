import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks

data = []

# Root of number of sites! Copy of this in runall, change both if changing either
nsites = [100, 130,  166,  216,  278,  360,  464,  600,  774, 1000] 
#nsites = [100,  112,  128,  144,  162,  184,  206,  234,  264,  298,  336,  380,  430,  484,546,  616,  696,  784,  886, 1000]

basefilename = "t3.txt"
for i in nsites:
    data.append(np.loadtxt("data/" + str(i) + "x" + str(i) +  basefilename, skiprows=1))

nfiles = len(data)
print(nfiles, "number of files")

nq = len(data[0][:,0])

data = np.array(data)


#################
# Calculations
#################

chi = np.array(nsites).astype(int)
logchi = np.log(chi)

minus_beta_del_nu = np.zeros(nq)
r = np.zeros(nq)

for i in range(nq):
    logpq = np.log(data[:,i,0])
    minus_beta_del_nu[i], _, r[i], _, _ = linregress(logchi, logpq) 

r2 = np.square(r)
idx_r2max = find_peaks(r2, prominence=0.01)[0][0]
pc = idx_r2max / (nq-1)
beta_del_nu = -minus_beta_del_nu[idx_r2max]
print("beta/nu", 0.1042, beta_del_nu)

idx_smax = np.argmax(data[:,:,1], axis=1)
smax = np.zeros(nfiles)
for i in range(nfiles):
    smax[i] = data[i,idx_smax[i], 1]
logsmax = np.log(smax)

gamma_del_nu = linregress(logchi, logsmax)[0]
print("gamma/nu", 1.7917, gamma_del_nu)

qmax = idx_smax / (nq-1)
x = np.log(pc-qmax)
nu = -linregress(x, logchi)[0]

gamma =  gamma_del_nu * nu
beta = beta_del_nu * nu

print("\tTheory\tNumerical")
print("Pc\t??\t", pc)
print("Beta\t", 5/36, "\t", beta)
print("Gamma\t", 43/18, "\t", gamma)
print("Nu\t", 4/3, "\t", nu)


#################
# Plotting
#################
plot = False

x = np.linspace(0, 1, nq)
lowlim = 0
highlim = 1
if plot:
    # Major component probability (p_inf)
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.title(r"$p_{\infty}$")
    for i in range(nfiles):
        y = data[i,:,0]
        plt.plot(x, y, label="N=%s" % nsites[i])
    plt.legend()

    # <s>
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.title("<s>")
    for i in range(nfiles):
        y = data[i,:,1]
        plt.plot(x,y, label="N=%s" % nsites[i])
    plt.legend()

    # chi
    plt.figure()
    plt.xlim(lowlim, highlim)
    plt.title(r"$\chi$")
    for i in range(nfiles):
        y = data[i,:,2]
        plt.plot(x,y, label="N=%s" % nsites[i])
    plt.legend()
    
    
plt.show()

