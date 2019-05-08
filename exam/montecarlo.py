import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import njit
from functools import lru_cache
from scipy.interpolate import UnivariateSpline

DEBUG = False


np.set_printoptions(linewidth=200)

np.random.seed(1)

#Lattice size
N = 10
M = 10
L = N*M

#Hamiltonian values
HA = 0.20
HB = 0.25

#Potential parameters
epsAA = 5.0
epsAB = 4.5
epsBB = 6

gamAA = 0.80
gamAB = 0.85
gamBB = 0.70

etaAA = 1.95
etaAB = 1.90
etaBB = 0.70


def V_create(a):
    VAA = epsAA * ( (gamAA/a)**6 - np.exp(-a/etaAA) )
    VAB = epsAB * ( (gamAB/a)**6 - np.exp(-a/etaAB) )
    VBB = epsBB * ( (gamBB/a)**6 - np.exp(-a/etaBB) )
    return [VAA, VAB, VBB]


def grid_create(xA):
    # Matrix of A and B
    # A = 1, B = 2
    nA = int(np.round(xA * L))
    # print(nA, "A atoms")
    grid = np.ones(L, dtype=int) * 2
    grid[0:nA] = 1
    # np.random.shuffle(grid)
    return grid


@njit # Substantial speed up
def H_create(grid, V):
    H = np.zeros((L,L))
    for i in np.arange(L):
        H[i,i] = (HA if grid[i] == 1 else HB)

        if i % M == 0:
            pot = V[(grid[i]*grid[i+M-1])//2]
            H[i, i+M-1] = pot 
            H[i+M-1, i] = pot

        else:
            pot = V[(grid[i-1]*grid[i])//2]
            H[i-1, i] = pot
            H[i, i-1] = pot

        pot = V[(grid[i]*grid[(i+M)%L])//2]
        H[i, (i+M)%L] = pot

        pot = V[(grid[i]*grid[(i-M)%L])//2]
        H[i, (i-M)%L] = pot
    return H



def swap(grid):
    swap1 = 0
    swap2 = 0
    while grid[swap1] == grid[swap2]:
        swap1, swap2 = np.random.randint(L, size=2)
    grid[swap1], grid[swap2] = grid[swap2], grid[swap1] 
    return grid



def F_calc(grid, V, kT):
    H = H_create(grid, V)
    E = np.linalg.eigvalsh(H)
    return -kT * np.sum( np.log(1 + np.exp(-E/kT) ) )



def update_grid_find_F(V, kT, grid, timesteps):

    # Flist = np.zeros(len(timesteps)) # Debugging
    Fprev = 0.0

    n_outer = 20 # How many different T-values
    n_inner = timesteps // n_outer # Iterations per T-value

    Tlist = np.logspace(-1, 3, n_outer)

    for i in range(n_outer):
        rand_compare_W = np.random.rand(n_inner)
        for j in range(n_inner):
            oldgrid = grid.copy()
            grid = swap(grid)
            
            F = F_calc(grid, V, kT)
            accept = True
            if F > Fprev:
                W = np.exp( -(F-Fprev) * Tlist[i] )
                accept = (W > rand_compare_W[j])

            # Swap back if bigger
            if accept:
                # Flist[i*n_inner + j] = F # Debugging
                Fprev = F

            else:
                # Swap back to old grid when condition is not met
                grid = oldgrid

    # DEBUG
    # plt.figure()
    # plt.plot(np.arange(timesteps), Flist, linestyle="", marker=".")
    return grid, F 



@lru_cache(maxsize=32) # Returns cached value if same input parameters are used again (avoid reiteration)
def enthalpy_start(a, kT):
    """
    Finds the potential values for a specific "a" and free energy for the two pure lattices
    """
    V = V_create(a)

    # FA
    xA_A = 1
    grid = grid_create(xA_A)
    FA = F_calc(grid, V, kT)
    
    # FB
    xA_B = 0
    grid = grid_create(xA_B)
    FB = F_calc(grid, V, kT)

    return V, FA, FB


    
def enthalpy_problem(timesteps_init, xAlist, a, kT, speedup=False):
    
    V, FA, FB = enthalpy_start(a, kT)

    N_xA = len(xAlist)
    Fs = np.zeros(N_xA)
    grids = np.zeros((N_xA, N*M))
    
    for i, xA in enumerate(xAlist):
        if speedup:
            timesteps = 100 + int(timesteps_init * 4*min(xA**2, (1-xA)**2))
        else:
            timesteps = timesteps_init
        startgrid = grid_create(xA)

        grids[i], Fs[i] = update_grid_find_F(V, kT, startgrid, timesteps)

        # Debugging
        if speedup:
            print("xA", xA, "F", Fs[i], "timesteps", timesteps)

    Fdeltas = Fs - FA*xAlist - FB*(1-xAlist)
    np.save("Fdeltas", Fdeltas)

    return Fdeltas, grids



def snapshot():
    timesteps = 20000

    # a = 1.3 # Aangstroem
    a = 0.9
    kT = 0.5 # 1/beta

    xAlist = np.array([0.25, 0.5, 0.75])

    start = time()
    _, grids = enthalpy_problem(timesteps, xAlist, a, kT)
    end = time()
    print("Snapshot time", end-start)

    plt.figure()
    plt.title("a = %1.1f, kT = %1.1f" % (a, kT))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.matshow(grids[i].reshape((N, M)), fignum=False)
        plt.gca().set_title(r"$x_A$ = %1.2f" % xAlist[i])
        plt.gca().xaxis.set_ticks_position('bottom')

    plt.savefig("figures/snapshot_a09k05.pdf")


def multirun():
    timesteps = 10000

    a = 1.3 # Aangstroem
    # a = 0.9
    kT = 0.5 # 1/beta

    Nxa = 98
    # Nxa = 32
    xAlist = np.linspace(0.01, 0.99, Nxa)

    start = time()
    Fdeltas, _ = enthalpy_problem(timesteps, xAlist, a, kT, speedup=True)
    end = time()
    print("Multirun time", end-start)

    plt.figure()
    plt.plot(xAlist, Fdeltas)
    plt.title("Enthalpy for values: a = %1.1f, kT = %1.1f" % (a, kT))
    plt.xlabel(r"$x_A$")
    plt.ylabel(r"$\Delta F(a,x_A)$")
    plt.savefig("figures/enthalpy_a%2.0f_kT%1.0f.pdf" % (a*10, kT*10))


def error_benchmark(a, kT):

    n_times = 5
    timelist = 20000 * np.ones(n_times, dtype=int)
    xA = np.array([0.5]) # Array because enthalpy_problem() requires a list
    Fdeltas = np.zeros(n_times) 

    for i, timesteps in enumerate(timelist):
        Fdeltas[i], _ = enthalpy_problem(timesteps, xA, a, kT)
        # print('Fdelta', Fdeltas[i])

    maxerr = 0
    for i in range(n_times-1):
        err = np.abs( (Fdeltas[i] - Fdeltas[i+1])) #/ Fdeltas[i])
        if err > maxerr:
            maxerr = err 

    return maxerr

        

def error_finder(timesteps, max_error, xA, a, kT):
    """
    Only accept smaller error until enough steps have been taken
    """
    error = 0
    counter = 0
    iterations_to_pass = 10
    
    Fdelta_prev = enthalpy_problem(timesteps, np.array([xA]), a, kT)[0][0]

    while error < max_error and counter < iterations_to_pass:
        counter += 1
        Fdelta = enthalpy_problem(timesteps, np.array([xA]), a, kT)[0][0]
        # print("Fdelta", Fdelta, "Fdelta_prev", Fdelta_prev)
        error = np.abs( (Fdelta - Fdelta_prev))# / Fdelta)
        Fdelta_prev = Fdelta

    # print("timesteps", timesteps, "error", error, "max_error", max_error)

    return error



min_misses = 0 # See how many has failed on the lower end
max_misses = 0 # --"-- higher end
def find_necessary_timesteps(xA, max_error, a, kT):
    """
    Finds the necessary Monte Carlo (time)steps needed 
    to guarantee a smaller error than max_error for a particular xA-value
    """
    ### Fetch globals to update
    global min_misses
    global max_misses

    ### Constants/settings
    minsteps = 100 # If timesteps < minsteps, accept current error and move on
    maxsteps = 10000 # If timesteps > maxsteps, accept current error and move on
    timesteps = 100
    prevsteps = 0
    preverror = 0
    found_top_value = False # True when it finds the first (biggest) timestep with small enough error

    print("Find necessary timesteps for xA =", xA)

    # Main event, returns when finding, or failing to find, an error
    while True:
        error = error_finder(timesteps, max_error, xA, a, kT)

        finished = (found_top_value and error > max_error)
        too_few_steps = timesteps < minsteps

        if finished or too_few_steps:
            if timesteps < minsteps and error > max_error: 
                min_misses += 1
            print("Error < maxerror:", preverror, max_error) # Debug
            return prevsteps # Latest result not good enough, return previous

        elif error < max_error:
            preverror = error # Remove later, only needed for debugging
            found_top_value = True
            prevsteps = timesteps
            timesteps = int(timesteps * 0.9)

        elif timesteps > maxsteps:
            max_misses += 1
            print("Error < maxerror:", preverror, max_error) # Debug
            return timesteps # Return this result, even though it is not good enough. Saves alot of time... 

        else:
            timesteps = int(timesteps*1.5)




def convergence_data_miner():
    """
    Runs through find_necessary_timesteps() for all xA.
    a and kT can be varied for different plots
    """

    ### Constants
    # a = 1.3 # Aangstroem
    a = 0.9
    kT = 0.1 # 1/beta

    ### Max error
    max_error = error_benchmark(a, kT)
    print(max_error)
    # max_error = 0.7
    # print("Max error:", max_error)

    ### Iteration settings
    iterations = 98
    timestep_list = np.zeros(iterations) 
    xA_list = np.linspace(0.01, 0.99, iterations)
    
    ### Main event and timing
    start = time()
    for i, xA in enumerate(xA_list):
        timesteps = find_necessary_timesteps(xA, max_error, a, kT)
        print()
        print("xA=%.2f needs %d timesteps" % (xA, timesteps))
        timestep_list[i] = timesteps
    end = time()
    print("Convergence miner time:", end - start)
    
    ### Save
    astr = ("%1.1f" % a)
    np.save("timesteps" + astr, timestep_list)
    # timestep_list = np.load("timesteps.npy")
    # timestep_list = np.load("timesteps" + astr + ".npy")
    
    ### How much failed?
    print("Min misses:", min_misses)
    print("Max misses:", max_misses)

    ### Plot
    plt.figure()
    plt.plot(xA_list, timestep_list, 'bx')
    plt.xlabel(r"$x_A$")
    plt.ylabel("Number of Monte Carlo steps")



if __name__ == "__main__":

    ### A few runs to create snapshots, quick
    # snapshot()

    ### Many runs to create enthalpy as a function of xA, slow
    multirun()

    ### Find necessary timesteps for different xA, extremely slow
    ### Needs some tweaking, doesn't always work right out of the box
    # convergence_data_miner()

    ### Table data
    # print(V_create(1.3))
    # print(V_create(0.9))
    
    plt.show()













