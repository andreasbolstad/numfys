import numpy as np

from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

cmap = plt.get_cmap('viridis')

np.set_printoptions(linewidth=400)


class ParticleBox():

    def __init__(self, N=1000, NUM_EIGVALS=100, V=None, Psi0=None):
        self.N = N # Number of points for discretization
        self.NUM_EIGVALS = NUM_EIGVALS # Number of eigenvalues to get
        self.nlist = np.linspace(1, NUM_EIGVALS, NUM_EIGVALS, dtype=int) # All n's of different eigenvalues
    
        if V:
            self.V = V
        else:
            self.V = np.zeros(N)

        self.x = np.linspace(0, 1, N)
        self.dx = 1/(N-1)
        self.la, self.psi = self.set_eigvalsvecs()
        
        self.Psi0 = Psi0


    def set_eigvalsvecs(self):
        """
        Hamiltonian can be represented as such:
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
        e = -1 / self.dx**2 * np.ones(self.N-3) + self.V[2:-1] # Next to diagonal elements (symmetric)

        psi = np.zeros((self.N, self.NUM_EIGVALS)) # Array of egeinvectors, 2nd axis specifies which eigenvalue is used
        # numpy.eigh_tridiagonal computes eigvals and eigvecs using a symmetric tridiagonal matrix
        la, psi[1:-1] = eigh_tridiagonal(d, e, select='i', select_range=(0, self.NUM_EIGVALS-1))

        for i in range(self.NUM_EIGVALS):
            psi[:,i] = psi[:,i] / np.sqrt(trapz(np.square(psi[:,i]), dx=self.dx))
            if psi[1,i] < 0: # Invert y-coordinate if negative at x=0 (want sinx behavior, not -sinx)
                psi[:,i] *= -1
        return la, psi 


    @property
    def alphas(self):
        product = self.psi.T * self.Psi0 
        alphas = np.zeros(self.NUM_EIGVALS)
        for i, elem in enumerate(product):
            alphas[i] = trapz(elem, dx=self.dx)
        return alphas



def get_Psi(alphas, la, psi, t=0):
    coeffs = alphas * np.exp(-1j*la*t)
    Psi_components = coeffs * psi 
    return np.sum(Psi_components, axis=1)


def animate(pb, alphas=None, xmin=0, xmax=1, ymin=-2, ymax=2, nt=2000):
    if alphas is None:
        alphas = pb.alphas
    fig, ax = plt.subplots()
    line1, = ax.plot([], [])
    line2, = ax.plot([], [])

    def anim_init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return line1, line2

    def update(t):
        Psi = get_Psi(alphas, pb.la, pb.psi, t=t)
        Psi2 = Psi * np.conj(Psi)
        Psi2 = Psi2.real # Discard imaginary zeros
        #print(r"Total probability =", trapz(Psi2, x=x))
        line1.set_data(pb.x, Psi.real)
        line2.set_data(pb.x, Psi2) 
        return line1, line2,

    baseAnimation = FuncAnimation(fig, update, init_func=anim_init, 
            frames=np.linspace(0, 2*np.pi, nt), interval=40, blit=True)
    plt.show()



## 2.4

def lambda_plot():
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
    plt.show()


def wave_plot():
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
    plt.show()


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
    plt.show()



#### 2.5
def alpha_print_test():
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
    pb = ParticleBox()
    pb.Psi0 = np.sqrt(2)*np.sin(np.pi * pb.x)
    animate(pb)


def psi0_delta_test():
    pb = ParticleBox()
    alphas = pb.psi[pb.N//2, :]/np.sqrt(pb.NUM_EIGVALS)
    animate(pb, alphas=alphas, ymin=-20, ymax=20, nt=5000000)



###############
### TASK 3 ####
###############

def high_barrier():
    N = 1000
    NUM_EIGVALS = 5
    x = np.linspace(0, 1, N)

    V = np.zeros(N)
    V[N//3:2*N//3] = 10000

    la, psi = eigvalsvecs(N, NUM_EIGVALS, V)
    
    plt.figure()
    for n, p in enumerate(psi.T, start=1):
        plt.plot(x, p, label=r"$\lambda_%s$" % str(n))
    plt.legend()
    plt.show()
    
    Psi0 = 1 / np.sqrt(2) * (psi[:, 0] + psi[:, 2]) 

    alphas = np.zeros(NUM_EIGVALS)
    for i in range(NUM_EIGVALS):
        alphas[i] = alpha_calculate(psi[:,i], Psi0)
    Psi1 = get_Psi(alphas, la, psi)
    Psi2 = get_Psi(alphas, la, psi, t=np.pi/(la[2]-la[0]))
    plt.figure()
    plt.plot(x, Psi1)
    plt.figure()
    plt.plot(x, Psi2)
    plt.show()
    #wave_animate(x, alphas, la, psi)


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
    #high_barrier()



