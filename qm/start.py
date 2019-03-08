import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapz
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')

np.set_printoptions(linewidth=200)

def eigvalsvecs(N, NUM_EIGVALS):
    dx = 1.0/(N-1)

    # Computing eigvals and vecs without the boundaries to avoid singular matrix. 
    d = 2 / dx**2 * np.ones(N-2) # Diagonal elements
    e = -1 / dx**2 * np.ones(N-3) # Next to diagonal elements (symmetric)

    yy = np.zeros((N,NUM_EIGVALS)) # Array of egeinvectors, 2nd axis specifies which eigenvalue is used
    # numpy.eigh_tridiagonal computes eigvals and eigvecs using a symmetric tridiagonal matrix
    la, yy[1:-1] = eigh_tridiagonal(d, e, select='i', select_range=(0,NUM_EIGVALS-1))

    psi = yy.T # Transpose, more practical and efficient format in the following
    for i in range(NUM_EIGVALS):
        psi[i] = psi[i] / np.sqrt(trapz(np.square(psi[i]), dx=dx))
        if psi[i,1] < 0: # Invert y-coordinate if negative at x=0 (want sinx behavior, not -sinx)
            psi[i] *= -1
    return la, psi 


## 2.4

def lambda_plot():
    N = 1000 # Number of points for discretization
    NUM_EIGVALS = 100 # Number of eigenvalues to get

    nlist = np.linspace(1, NUM_EIGVALS, NUM_EIGVALS, dtype=int) # All n's of different eigenvalues
    ala = (nlist*np.pi)**2 # Analytical lambdas (eigenvalues)
    la, _ = eigvalsvecs(N, NUM_EIGVALS)

    plt.figure()
    plt.title(r'$\lambda_n(E_n) = \frac{E_n}{E_0},\ E_0 = \frac{\hbar^2}{2mL^2}$')
    plt.xlabel(r'n')
    plt.ylabel(r"$\lambda_n$")
    plt.plot(nlist, ala, label=r"$\lambda^{analytical}$")
    plt.plot(nlist, la, label=r"$\lambda^{numerical}$")
    plt.legend()


def wave_plot():
    PLOT_RANGE = [1, 2, 3, 4, 5] # Choose n's to plot corresponding psi_n's

    N = 100 # Number of points for discretization
    NUM_EIGVALS = max(PLOT_RANGE) # Number of eigenvalues to get

    _, psi = eigvalsvecs(N, NUM_EIGVALS)

    x = np.linspace(0, 1, N)
    apsi = np.empty((NUM_EIGVALS, N), dtype=float) 
    for n in range(1, NUM_EIGVALS+1):
        apsi[n-1] = np.sqrt(2) * np.sin(n*np.pi*x)
    
    plt.figure()
    plt.xlabel("x'")
    plt.ylabel(r"$\psi$")
    for n in PLOT_RANGE:
        i = n - 1
        color = cmap(float(i)/NUM_EIGVALS)
        plt.plot(x, psi[i], marker="x", c=color)
        plt.plot(x, apsi[i], label="E%s" % str(n), c=color)
        plt.legend()


def error_plot(n_eigval):
    N_list = np.array(range(20, 201))
    error = []
    for N in N_list:
        psi = eigvalsvecs(N, n_eigval)[1][n_eigval-1]
        x = np.linspace(0, 1, N)
        apsi = np.sqrt(2) * np.sin(n_eigval*np.pi*x) 
        abs_err = np.abs(psi-apsi)
        avg_err = np.sum(abs_err) / N 
        error.append(avg_err)
    plt.figure()
    plt.plot(N_list, error)



#### 2.5
def alpha_calculate(psi, Psi):
    product = psi * Psi
    dx = 1 / (psi.size - 1)
    return trapz(product, dx=dx)


def alpha_print_test():
    # Prints a grid of all combinations of inner products, from n=1 to NUM_EIGVALS
    N = 10
    NUM_EIGVALS = 5
    _, psi = eigvalsvecs(N, NUM_EIGVALS)
    dx = 1/(N-1)
    overlaps = np.zeros((NUM_EIGVALS, NUM_EIGVALS))
    for i in range(NUM_EIGVALS):
        for j in range(NUM_EIGVALS):
            overlaps[i,j] = alpha_calculate(psi[i], psi[j])
    print(overlaps)


if __name__ == "__main__":
    ## 2.4
    #lambda_plot()
    #wave_plot()
    #error_plot(3)

    ## 2.5
    alpha_print_test()

    ## 2.6
    

    plt.show()



