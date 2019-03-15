import numpy as np

from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

cmap = plt.get_cmap('viridis')

np.set_printoptions(linewidth=400)

class ParticleBox():               
    """
    Class for particle boxes, convenient for storing all relevant variables.
    
    Member variables:
    N, NUM_EIGVALS, V, Psi0, nlist, x, dx, la, psi, Psi0, alphas
    """


    def __init__(self, N=1000, NUM_EIGVALS=100, V=None, Psi0=None):
        self.N = N # Number of points for discretization
        self.NUM_EIGVALS = NUM_EIGVALS # Number of eigenvalues to get
        self.nlist = np.linspace(1, NUM_EIGVALS, NUM_EIGVALS, dtype=int) # All n's of different eigenvalues
    
        if V is not None:
            self.V = V
        else:
            self.V = np.zeros(N)

        self.x = np.linspace(0, 1, N)
        self.dx = 1/(N-1)
        self.la, self.psi = self.set_eigvalsvecs()
        
        self.Psi0 = Psi0


    def set_eigvalsvecs(self):
        """
        Hamiltonian can be represented similar to:
        --                                -- 
        |   2+V   -1                       |    
        |   -1    2+V   -1                 |    
        |         -1     .    .            |         
        |                .    .    .       |     
        |                     .    .    .  |     
        |                          .    .  |     
        --                                --    
        """
        # Computing eigvals and vecs without the boundaries to avoid singular matrix. 
        d = 2 / self.dx**2 * np.ones(self.N-2) + self.V[1:-1] # Diagonal elements
        e = -1 / self.dx**2 * np.ones(self.N-3) # Next to diagonal elements (symmetric)

        psi = np.zeros((self.N, self.NUM_EIGVALS)) # Array of egeinvectors, 2nd axis specifies which eigenvalue is used
        # numpy.eigh_tridiagonal computes eigvals and eigvecs using a symmetric tridiagonal matrix
        la, psi[1:-1] = eigh_tridiagonal(d, e, select='i', select_range=(0, self.NUM_EIGVALS-1))

        for i in range(self.NUM_EIGVALS):
            psi[:,i] = psi[:,i] / np.sqrt(trapz(np.square(psi[:,i]), dx=self.dx))
            if psi[1,i] < 0: # Invert y-coordinate if negative at x=0 (want sinx behavior, not -sinx)
                psi[:,i] *= -1
        return la, psi 


    @property # Call as a variable instead of function, i.e. obj.a instead of obj.a()
    def alphas(self):
        product = self.psi.T * self.Psi0 
        alphas = np.zeros(self.NUM_EIGVALS)
        for i, elem in enumerate(product):
            alphas[i] = trapz(elem, dx=self.dx)
        return alphas



def get_Psi(pb, alphas=None, t=0):
    """
    Calculate Psi at a particular time t
    If the argument alphas is given, use these values instead of alphas from pb (the particleBox instance)
    """
    if not type(alphas) is np.ndarray:
        alphas = pb.alphas
    coeffs = alphas * np.exp(-1j*pb.la*t)
    Psi_components = coeffs * pb.psi 
    return np.sum(Psi_components, axis=1)


def animate(pb, alphas=None, xmin=0, xmax=1, ymin=-2, ymax=2, nt=1000, tmax=2*np.pi):
    """
    Animates the probability distribution evolution of the wavefunction of pb (ParticleBox object)
    """
    fig, ax = plt.subplots()
    line1, = ax.plot([], [])

    def anim_init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return line1, 

    def update(t):
        Psi = get_Psi(pb, alphas=alphas, t=t)
        Psi2 = np.absolute(Psi)**2
        Psi2 = Psi2.real # Discard imaginary zeros
        #print(r"Total probability =", trapz(Psi2, x=x))
        line1.set_data(pb.x, Psi2) 
        return line1,

    _ = FuncAnimation(fig, update, init_func=anim_init, 
            frames=np.linspace(0, tmax, nt), interval=40, blit=True)



## 2.4

def lambda_plot():
    """
    Plots the eigenvalues/lambdas as a function of n
    1st plot: Numerical and analytical version
    2nd plot: Error (i.e. difference between the numerical and analytical version)
    """
    pb = ParticleBox()

    ala = (pb.nlist*np.pi)**2 # Analytical lambdas (eigenvalues)

    plt.figure()
    plt.title(r'$\lambda_n(E_n) = \frac{E_n}{E_0},\ E_0 = \frac{\hbar^2}{2mL^2}$')
    plt.xlabel(r'n')
    plt.ylabel(r"$\lambda_n$")
    plt.plot(pb.nlist, ala, label=r"$\lambda^{analytical}$")
    plt.plot(pb.nlist, pb.la, label=r"$\lambda^{numerical}$")
    plt.legend()

    plt.figure()
    plt.title("Error of lambda_n")
    plt.plot(pb.nlist, ala-pb.la)


def wave_plot():
    """
    Plots the wavefunctions corresponding to different eigenvalues/lambdas
    In figure: x-markers numerical, lines analytical
    """
    PLOT_RANGE = [1, 2, 3, 4, 5] # Choose n's to plot corresponding psi_n's

    N = 100 # Number of points for discretization
    NUM_EIGVALS = max(PLOT_RANGE) # Number of eigenvalues to get

    pb = ParticleBox(N, NUM_EIGVALS) 

    x = pb.x 
    apsi = np.empty((NUM_EIGVALS, N), dtype=float) 
    for n in range(1, NUM_EIGVALS+1):
        apsi[n-1] = np.sqrt(2) * np.sin(n*np.pi*x)
    
    plt.figure()
    plt.xlabel("x'")
    plt.ylabel(r"$\psi$")
    for n in PLOT_RANGE:
        i = n - 1
        color = cmap(float(i)/NUM_EIGVALS)
        plt.plot(x, pb.psi[:,i], marker="x", c=color)
        plt.plot(x, apsi[i], label="E%s" % str(n), c=color)
        plt.legend()


def error_plot(n_eigval):
    """
    Set 1st argument (n_eigval) to the eigenvalue you want to inspect the error of
    """
    N_list = np.array(range(20, 201))
    error = []
    for N in N_list:
        pb = ParticleBox(N, n_eigval)
        psi = pb.psi[:, n_eigval-1]
        x = np.linspace(0, 1, N)
        apsi = np.sqrt(2) * np.sin(n_eigval*np.pi*x) 
        abs_err = np.abs(psi-apsi)
        avg_err = np.sum(abs_err) / N 
        error.append(avg_err)
    plt.figure()
    plt.plot(N_list, error)



#### 2.5
def alpha_print_test():
    """
    Prints out a table of alphas, i.e. scalar products of the eigenvectors
    Should equal 0 for differing eigenvectors and 1 for identical eigenvectors
    """
    N = 10
    NUM_EIGVALS = 5
    pb = ParticleBox(N=N, NUM_EIGVALS=NUM_EIGVALS)

    ## Unimportant formatting
    print("Square of absolute values")
    nums = list(range(1, NUM_EIGVALS+1))
    text = "\t\t" + "\t\t".join([str(n) for n in nums])
    print(text)
    ##

    for n in pb.nlist:
        i = n-1
        pb.Psi0 = pb.psi[:, i]
        print(n, pb.alphas)


### 2.6

def psi0_sine_test():
    """
    Animates the time evolution of |psi|^2 with the initial wavefunction being a sine funciton
    """
    pb = ParticleBox()
    pb.Psi0 = np.sqrt(2)*np.sin(np.pi * pb.x)
    animate(pb)


def psi0_delta_test():
    """
    Animates the time evolution of |psi|^2 with the initial wavefunction being a delta funciton
    """
    pb = ParticleBox()
    alphas = pb.psi[pb.N//2, :]/np.sqrt(pb.NUM_EIGVALS)
    animate(pb, alphas=alphas, ymin=-20, ymax=20, tmax=1e-2)



###############
### TASK 3 ####
###############

## 3.1
def high_barrier():
    """
    A collective function for several procedures:
    1. Create a particle box with a high barrier
    2. Plot some of the first eigenfunctions
    3. Animate the time evolution of a prob. distr., with i.v. as a l.c. of psi1 and psi2
    4. Plot the same prob. distr. at times t=0 and t=pi/(l2-l1)
    """
    N = 900 # Must be a multiple of 3!!!
    NUM_EIGVALS = 10 


    v0 = 1000
    V = np.zeros(N)
    V[N//3:2*N//3] = v0

    pb = ParticleBox(N=N, NUM_EIGVALS=NUM_EIGVALS, V=V)
    
    plt.figure()
    for n, p in enumerate(pb.psi.T[0:2], start=1):
        plt.plot(pb.x, p.real, label=r"$\lambda_%s$" % str(n))
    plt.legend()
   
    pb.Psi0 = 1 / np.sqrt(2) * (pb.psi[:, 0] + pb.psi[:,1])  
    animate(pb, ymin=-4, ymax=10, tmax=10*np.pi/(pb.la[1]-pb.la[0]))

    Psi1 = get_Psi(pb, t=0)
    Psi2 = get_Psi(pb, t=np.pi/(pb.la[1]-pb.la[0]))

    print("lambdas:", pb.la[:10])
    
    plt.figure()
    plt.plot(pb.x, V/max(V))
    plt.plot(pb.x, np.absolute(Psi1)**2, label="t=0", color='b')

    plt.figure()
    plt.plot(pb.x, V/max(V))
    plt.plot(pb.x, np.absolute(Psi2)**2, label="t=pi/(l2-l1)")



## 3.2

def root_finding():
    """
    func() : Calculates f for a particular value of lambda

    plot_f() : Plot the function f of lambda (to get a rough idea of where the roots are)

    find_roots() : 
     - Find minima of the intervals found from inspection of the plot from plot_f
     - Find root left and right of each minima (and print to console)
    """

    def func(la):
        k = np.sqrt(la)
        K = np.sqrt(v0 - la)
        Ksin = K * np.sin(k/3)
        kcos = k * np.cos(k/3)
        return np.exp(K/3) * np.square(Ksin + kcos) - np.exp(-K/3) * np.square(Ksin - kcos)


    def find_roots(f):
        mins = np.array(find_peaks(-f)[0]) * v0 / N
        zeros = []
        d = 30
        for c in mins:
            a = max(c - d, c*0.5)
            b = min(c + d, v0)
            if func(a) > 0:
                a = c * 0.5
            if func(a) > 0 and func(c) < 0:
                zeros.append( brentq( func, a, c ) )
            if func(b) > 0 and func(c) < 0:
                zeros.append( brentq( func, c, b ) )
        return zeros


    N = 900000
    v0 = 1000
    la_list = np.linspace(0, v0, N)
    f_la = func(la_list) 

    # 3.5, also need to compare to 3.4
    print(find_roots(f_la))
    plt.figure()
    plt.plot(la_list[N//2:], f_la[N//2:])

    # 3.6
    niter = 10
    num_zeros = []
    v0s = np.linspace(50, 2000, niter)
    for v0 in v0s: 
        la_list = np.linspace(0, v0, N)
        f_la = func(la_list)
        num_zeros.append(len(find_roots(f_la))) 

    plt.figure()
    plt.plot(v0s, num_zeros)


###############
### Main ######
###############
if __name__ == "__main__":
    ## 2.4
    #lambda_plot()
    #wave_plot()
    #error_plot(3)

    ## 2.5
    #alpha_print_test()

    ## 2.6
    #psi0_sine_test()
    #psi0_delta_test()

    ## 3.1
    #high_barrier()
    
    ## 3.2
    root_finding()

    plt.show()
