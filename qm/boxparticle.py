# Imports
from functools import partial
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv

# Settings
cmap = plt.get_cmap('viridis')
np.set_printoptions(linewidth=400)


def abs_squared(psi):
    return psi.real**2 + psi.imag**2


class ParticleBox():               
    """
    Class for particle boxes, convenient for storing all relevant variables.
    
    Member variables:
    N, NUM_EIGVALS, V, Psi0, nlist, x, dx, la, psi, Psi0, alphas
    """

    def __init__(self, N=1002, NUM_EIGVALS=100, v0=0, Psi0=None):
        self.N = N - N%3 # Number of points for discretization, muliple of 3 to get symmetric wells.
        self.NUM_EIGVALS = NUM_EIGVALS # Number of eigenvalues to get
        self.nlist = np.linspace(1, NUM_EIGVALS, NUM_EIGVALS, dtype=int) # All n's of different eigenvalues
    
        self.V = np.zeros(self.N)
        self.V[self.N//3 : 2*self.N//3] = v0

        self.x = np.linspace(0, 1, self.N)
        self.dx = 1/(self.N-1)
        
        self.d = 2 / self.dx**2 * np.ones(self.N) + self.V # Diagonal elements
        self.d[0] = 0
        self.d[-1] = 0

        self.e = -1 / self.dx**2 * np.ones(self.N-1) # Off-diagonal elements (symmetric)
        self.e[0] = 0
        self.e[-1] = 0
        
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

        psi = np.zeros((self.N, self.NUM_EIGVALS), dtype=np.complex_) # Array of egeinvectors, 2nd axis specifies which eigenvalue is used
        # numpy.eigh_tridiagonal computes eigvals and eigvecs using a symmetric tridiagonal matrix
        la, psi[1:-1] = eigh_tridiagonal(self.d[1:-1], self.e[1:-1], select='i', select_range=(0, self.NUM_EIGVALS-1))

        for i in range(self.NUM_EIGVALS):
            psi[:,i] = psi[:,i] / np.sqrt(trapz(np.square(psi[:,i]), dx=self.dx))
            if psi[1,i] < 0: # Invert y-coordinate if negative at x=0 (want sinx behavior, not -sinx)
                psi[:,i] *= -1
        return la, psi 


    @property # Call as a variable instead of function, i.e. obj.a instead of obj.a()
    def alphas(self):
        product = self.psi.T * self.Psi0 
        bracket = product.real
        alphas = np.zeros(self.NUM_EIGVALS)
        for i, elem in enumerate(bracket):
            alphas[i] = trapz(elem, dx=self.dx)
        return alphas


# Mostly animation functions:
def psi_at_time_box(t, pb, alphas):
    """Return x and psi at a particlar time, using linear combination of alphas"""
    coeffs = alphas * np.exp(-1j*pb.la*t)
    Psi_components = coeffs * pb.psi 
    return pb.x, np.sum(Psi_components, axis=1)


def psi_at_time_box_noalphas(t, pb):
    """Convenience function, use pb.alphas instead of manually specifying them"""
    return psi_at_time_box(t, pb, pb.alphas)


def psi_at_time_crank(t, x, psi, A):
    psi[:] = A*psi
    return x, psi


def animate(func_x_psi, *fargs, xmin=0, xmax=1, ymin=-2, ymax=2, nt=1000, tmax=2*np.pi):
    """
    Animates the time evolution of the wavefunction of pb (ParticleBox object)
    line1 : Represents the real value of the wavefunction
    line2 : Represents the absolute square value of the wavefunction
    """
    fig, ax = plt.subplots()
    line1, = ax.plot([], [])
    line2, = ax.plot([], [])

    def anim_init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return line1, line2,

    def update(t):
        args = fargs
        x, Psi = func_x_psi(t, *fargs)
        Psi2 = abs_squared(Psi)
        line1.set_data(x, Psi.real)
        line2.set_data(x, Psi2) 
        return line1, line2,

    _ = FuncAnimation(fig, update, init_func=anim_init, 
            frames=np.linspace(0, tmax, nt), interval=40, blit=True)
    plt.show()



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

    apsi = np.empty((NUM_EIGVALS, pb.N), dtype=float) 
    for n in range(1, NUM_EIGVALS+1):
        apsi[n-1] = np.sqrt(2) * np.sin(n*np.pi*pb.x)
    
    plt.figure()
    plt.xlabel("x'")
    plt.ylabel(r"$\psi$")
    for n in PLOT_RANGE:
        i = n - 1
        color = cmap(float(i)/NUM_EIGVALS)
        plt.plot(pb.x, pb.psi[:,i], marker="x", c=color)
        plt.plot(pb.x, apsi[i], label="E%s" % str(n), c=color)
        plt.legend()


def error_plot(n_eigval):
    """
    Set 1st argument (n_eigval) to the eigenvalue you want to inspect the error of
    """
    N_list = np.array(range(20, 201))
    error = []
    for n in N_list:
        pb = ParticleBox(n, n_eigval)
        psi = pb.psi[:, n_eigval-1]
        apsi = np.sqrt(2) * np.sin(n_eigval*np.pi*pb.x) 
        abs_err = np.abs(psi-apsi)
        avg_err = np.sum(abs_err) / pb.N 
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
    animate(psi_at_time_box_noalphas, pb)
    plt.show()


def psi0_delta_test():
    """
    Animates the time evolution of |psi|^2 with the initial wavefunction being a delta funciton
    """
    pb = ParticleBox()
    alphas = pb.psi[pb.N//2, :]/np.sqrt(pb.NUM_EIGVALS)
    animate(psi_at_time_box, pb, alphas, ymin=-20, ymax=20, tmax=1e-2)



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

    pb = ParticleBox(N=N, NUM_EIGVALS=NUM_EIGVALS, v0=v0)
    
    plt.figure()
    for n, p in enumerate(pb.psi.T[0:2], start=1):
        plt.plot(pb.x, p.real, label=r"$\lambda_%s$" % str(n))
    plt.legend()
   
    pb.Psi0 = 1 / np.sqrt(2) * (pb.psi[:, 0] + pb.psi[:,1])  
    animate(psi_at_time_box_noalphas, pb, ymin=-4, ymax=10, tmax=10*np.pi/(pb.la[1]-pb.la[0]))
    
    t1 = 0
    x, Psi1 = psi_at_time_box_noalphas(t1, pb)
    

    t2 = np.pi/(pb.la[1]-pb.la[0])
    x, Psi2 = psi_at_time_box_noalphas(t2, pb)

    print("lambdas:", pb.la[:10])
    
    y = pb.V/max(pb.V)

    plt.figure()
    plt.plot(x, y)
    plt.plot(pb.x, abs_squared(Psi1), label="t=0", color='b')

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, abs_squared(Psi2), label="t=pi/(l2-l1)")



## 3.2

def root_finding():
    """  
    func() : Calculates f for a particular value of lambda

    plot_f() : Plot the function f of lambda (to get a rough idea of where the roots are)

    find_roots() : 
     - Find minima of the intervals found from inspection of the plot from plot_f
     - Find root left and right of each minima (and print to console)
    """

    def func(la, v0):
        k = np.sqrt(la)
        K = np.sqrt(v0 - la)
        Ksin = K * np.sin(k/3)
        kcos = k * np.cos(k/3)
        return np.exp(K/3) * np.square(Ksin + kcos) - np.exp(-K/3) * np.square(Ksin - kcos)


    def find_roots(f, v0):
        mins = np.array(find_peaks(-f)[0]) * v0 / N
        zeros = []
        d = 30
        for c in mins:
            a = max(c - d, c*0.5)
            b = min(c + d, v0)

            c = minimize_scalar(func, args=v0, bounds=(a,b), method='bounded').x 

            if func(a, v0) > 0 and func(c, v0) < 0:
                zeros.append( brentq( func, a, c, args=v0) )
            if func(b, v0) > 0 and func(c, v0) < 0:
                zeros.append( brentq( func, c, b, args=v0 ) )
        return zeros


    ### TESTS

    N = 90000
    v0 = 1000
    la_list = np.linspace(0, v0, N)
    f_la = func(la_list, v0) 

    # 3.5
    def print_roots():
        print("Roots/zeros:", find_roots(f_la, v0)) # Compare with 3.4


    def plot_f():
        plt.figure()
        plt.plot(la_list, f_la)

    # 3.6
    def plot_number_of_zeros():
        niter = 1000
        num_zeros = []
        v0s = np.linspace(10, 2500, niter)
        print("Plotting... Should take less than a minute.")
        for v0 in v0s:
            la_list = np.linspace(0, v0, N)
            f_la = func(la_list, v0)
            num_zeros.append(len(find_roots(f_la, v0))) 

        plt.figure()
        plt.plot(v0s, num_zeros)

    def find_small_eig_v0():
        v0 = 20
        pnr = 0
        multiplier = 1
        while abs(multiplier) > 1e-5:
            la_list = np.linspace(0, v0, N)
            f_la = func(la_list, v0)
            nnr = len(find_roots(f_la, v0))
            v0 += 1 * multiplier
            if nnr != pnr:
                pnr = nnr
                multiplier /= -10
        print("Smallest value of v0 that gives 1 eigenvalue < v0:", v0)

    # Each test is independent. You can run only one if you want
    print_roots()
    plot_f()
    plot_number_of_zeros()
    find_small_eig_v0()


## 3.3
def timestepping_euler():
    """
    A simple test function, not well implemented as is...
    Vary N and dt to test performance of the forward euler method on the
    Schrodinger equation.
    Testing for different values will show how divergence does/does not
    correspond to CFL number
    Plots values after nt number of iterations. Reduce this number when
    increasing time step
    """
    N = 999
    NUM_EIGVALS = 10 

    v0 = 1000

    pb = ParticleBox(N=N, NUM_EIGVALS=NUM_EIGVALS, v0=v0)
    psi_t = pb.psi[:,0]

    H = diags((pb.d, pb.e, pb.e), (0, -1, 1))
    
    dt = 1e-6
    nt = 26

    dx = pb.dx
    cfl = dt/dx**2
    print("dt=%s, dx=%s, cfl=%s" % (str(dt), str(dx), str(cfl)))

    f, ax = plt.subplots(1)
    for i in range(nt):
        psi_t = psi_t - 1j*dt*H.dot(psi_t)
    ax.plot(pb.x, psi_t.real, label="Real")
    ax.plot(pb.x, psi_t.imag, label="Imag")
    f.legend()


def timestepping_crank():
    """Testing stuff with the Crank Nicolson scheme given in eq. 3.8 in the
    assignment."""
    N = 999
    NUM_EIGVALS = 1
    v0 = 1000 

    pb = ParticleBox(N=N, NUM_EIGVALS=NUM_EIGVALS, v0=v0)
    psi_t = pb.psi[:,0]
    H = diags((pb.d, pb.e, pb.e), (0, -1, 1))

    dt = 0.01

    I = eye(N)
    R = I - 1j/2*dt*H
    L = I + 1j/2*dt*H
    Linv = inv(L)

    A = R * Linv
    
    animate(psi_at_time_crank, pb.x, psi_t, A, ymax=4)


## 4.1
def lower_barrier():
    v0 = 100
    vr = 50
    



###############
### Main ######
###############
if __name__ == "__main__":
    """
    Numbers indicate which section (not task) in the assignment the function calls are relevant for
    """

    ## 2.4
    # lambda_plot()
    # wave_plot()
    # error_plot(3)

    ## 2.5
    # alpha_print_test()

    ## 2.6
    # psi0_sine_test()
    # psi0_delta_test()

    ## 3.1
    high_barrier()
    
    ## 3.2
    # root_finding()

    ## 3.3
    # timestepping_euler()
    # timestepping_crank()

    plt.show()
    




