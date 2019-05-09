"""
The program is structured as a few sections:
1. Settings
2. Constants
3. Base functions
4. Monte Carlo / Metropolis algorithm
5. Enthalpy
6. Convergence test functions
7. Functions called by main
8. MAIN
"""



############
# Settings
############

import numpy as np # Numerics
import matplotlib.pyplot as plt # Plotting
from time import time # Measure runtimes
from numba import njit # Decorator for precompiling 
from functools import lru_cache # Cache previous function calls


np.set_printoptions(linewidth=200) # For printing large arrays, Hamiltonian

np.random.seed(1) # Initialize pseudorandom number generator



##############
# Constants
##############

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






##################
# Base functions
##################

def V_create(a):
    """Returns an array of the nearest neighbour potentials"""
    VAA = epsAA * ( (gamAA/a)**6 - np.exp(-a/etaAA) )
    VAB = epsAB * ( (gamAB/a)**6 - np.exp(-a/etaAB) )
    VBB = epsBB * ( (gamBB/a)**6 - np.exp(-a/etaBB) )
    return [VAA, VAB, VBB]


def grid_create(xA):
    """
    Creates a grid of length L=M*N to represent the 2D lattice.
    1 represents atom type A, 2 represents B (why is explained in H_create())    
    """
    nA = int(np.round(xA * L))
    grid = np.ones(L, dtype=int) * 2
    grid[0:nA] = 1
    # np.random.shuffle(grid) # Not needed
    return grid



@njit # Substantial speed up, x10-100 for for-loops
def H_create(grid, V):
    """
    Creates the Hamiltonian for the lattice

    Because the atom types are marked 1 and 2, we can do a nifty indexing trick.
    Multiplying two values from the grid will give 1, 2 or 4. By using integer division of 2,
    we get 0, 1 or 2 respectively. Since V = [VAA, VAB, VBB], the oppropriate potential can be
    picked out directly with calculating indices instead of using if statements.
    """
    H = np.zeros((L,L))
    for i in np.arange(L):
        H[i,i] = (HA if grid[i] == 1 else HB)

        ## Vx
        if i % M == 0:
            Vx = V[ (grid[i]*grid[i+M-1]) // 2 ]
            H[i, i+M-1] = Vx 
            H[i+M-1, i] = Vx

        else:
            Vx = V[ (grid[i-1]*grid[i]) // 2 ]
            H[i-1, i] = Vx
            H[i, i-1] = Vx

        ## Vy
        Vy = V[ (grid[i]*grid[(i+M)%L]) // 2 ]
        H[i, (i+M) % L] = Vy

        Vy = V[ (grid[i]*grid[(i-M)%L]) // 2 ]
        H[i, (i-M) % L] = Vy

    return H



def swap(grid):
    """Chooses indices randomly. If the indices point to atoms of the same type, start over."""
    swap1 = 0
    swap2 = 0
    while grid[swap1] == grid[swap2]: # Same kind? Always true first time since both indices are the same
        swap1, swap2 = np.random.randint(L, size=2) # Two random numbers from 0 up to, but not including, L
    grid[swap1], grid[swap2] = grid[swap2], grid[swap1] 
    return grid # New grid with two atoms swapped




######################################
# Monte Carlo / Metropolis algorithm
######################################

def update_grid_find_F(V, kT, grid, timesteps):
    """
    Monte Carlo scheme based on the Metropolis algorithm
    Returns the minimized free energy F and the corresponding grid/lattice

    Uncommenting the "debugging lines" will create a plot showing the development of F for every step
    """

    # Flist = np.zeros(len(timesteps)) # Debugging
    Fprev = 0.0

    n_outer = 20 # How many different T-values
    n_inner = timesteps // n_outer # Iterations per T-value

    Tlist = np.logspace(-1, 3, n_outer)

    for i in range(n_outer):
        # Random numbers to be compared with acceptance criterium 
        rand_compare_W = np.random.rand(n_inner) # should probably be moved inside "if F > Fprev"
        for j in range(n_inner):
            oldgrid = grid.copy()
            grid = swap(grid)
            
            F = F_calc(grid, V, kT)
            accept = True
            ### Metropolis algorithm
            if F > Fprev:
                W = np.exp( -(F-Fprev) * Tlist[i] )
                accept = (W > rand_compare_W[j])

            # Swap back if random number is smaller than W
            if accept:
                Fprev = F
                # Flist[i*n_inner + j] = F # Debugging

            else:
                # Swap back to old grid when condition is not met
                grid = oldgrid

    ### DEBUG
    # plt.figure()
    # plt.plot(np.arange(timesteps), Flist, linestyle="", marker=".")
    return grid, F 







###########
# Enthalpy 
###########

def F_calc(grid, V, kT):
    """
    Returns the free energy F for a specific kT 
    """
    H = H_create(grid, V) # Create the Hamiltonian for specific grid and V (potential)
    E = np.linalg.eigvalsh(H) # Calculate eigenergies. Using np.linalg.eigvalsh since H is symmetric
    return -kT * np.sum( np.log(1 + np.exp(-E/kT) ) ) # Calculate F 



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
    """
    Returns the minimized enthalpies and corresponding grids for each xA in xAlist
    If speedup is true, then timesteps are moderated to save time 
    """
    ### Find neighbour potentials, F for only A atoms and F for only B atoms 
    V, FA, FB = enthalpy_start(a, kT)

    ### Init
    N_xA = len(xAlist)
    Fs = np.zeros(N_xA)
    grids = np.zeros((N_xA, N*M))
    
    for i, xA in enumerate(xAlist):
        ### Choose number of timesteps 
        if speedup:
            timesteps = 100 + int(timesteps_init * 2*min(xA, (1-xA))) # Linear moderation
            # timesteps = 100 + int(timesteps_init * 4*min(xA**2, (1-xA)**2)) # Square moderation
        else:
            timesteps = timesteps_init
        
        ### Create grid
        startgrid = grid_create(xA)

        ### Find final grid and minimized free energy F
        grids[i], Fs[i] = update_grid_find_F(V, kT, startgrid, timesteps)

        ### Debugging
        if speedup:
            print("xA", xA, "F", Fs[i], "timesteps", timesteps)

    ### Find enthalpy
    Fdeltas = Fs - FA*xAlist - FB*(1-xAlist)

    ### Save
    np.save("Fdeltas", Fdeltas)

    return Fdeltas, grids






##############################
# Convergence test functions
##############################

def error_benchmark(a, kT):
    """
    Do a few runs for x_A=0.5 and find the largest error
    (Can be improved alot, but it is not the bottleneck of the convergence test at the time of writing this)
    """
    
    n_times = 15
    timesteps = 10000
    xA = np.array([0.5]) # Array because enthalpy_problem() requires a list

    Fdeltas = np.zeros(n_times) 
    for i in range(n_times):
        Fdeltas[i], _ = enthalpy_problem(timesteps, xA, a, kT)

    maxerr = 0
    for i in range(n_times-1):
        err = np.abs( (Fdeltas[i] - Fdeltas[i+1])) 
        if err > maxerr:
            maxerr = err 

    return maxerr

        

def error_finder(timesteps, max_error, xA, a, kT):
    """
    Only accept smaller error until enough steps have been taken
    1. Find F
    2. Calculate difference from previous F (absolute error)
    3. If error > max_error return, otherwise continue until all iterations have passed and return
    """
    error = 0
    counter = 0
    iterations_to_pass = 10
    
    Fdelta_prev = enthalpy_problem(timesteps, np.array([xA]), a, kT)[0][0]

    while error < max_error and counter < iterations_to_pass:
        counter += 1
        Fdelta = enthalpy_problem(timesteps, np.array([xA]), a, kT)[0][0]
        error = np.abs( (Fdelta - Fdelta_prev))
        Fdelta_prev = Fdelta

    # print("timesteps", timesteps, "error", error, "max_error", max_error)

    return error



min_misses = 0 # See how many has failed on the lower end
max_misses = 0 # --"-- higher end
def find_necessary_timesteps(xA, max_error, a, kT):
    """
    Finds the necessary Monte Carlo (time)steps to guarantee a smaller error than max_error for a particular xA-value
    1. Increase timesteps until a error_finder() returns an error < max_error
    2. Reduce timesteps until error_finder() returns an error > max_error
    3. Return number of timesteps for the last run with error < max_error
    Alt. Exit because of too few (avoid crash) or too many timesteps (avoid spending infinite time)
    """
    ### Fetch globals
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






###########################
# Functions called by main
###########################

def snapshot(a=1.3, kT=0.1):
    timesteps = 20000
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



def multirun(a=1.3, kT=0.1):
    ### Init
    timesteps = 20000
    Nxa = 98 # Number of xA-values
    xAlist = np.linspace(0.01, 0.99, Nxa)

    ### Monte Carlo
    start = time()
    Fdeltas, _ = enthalpy_problem(timesteps, xAlist, a, kT, speedup=True)
    end = time()
    print("Multirun time", end-start)

    ### Plot
    plt.figure()
    plt.plot(xAlist, Fdeltas)
    plt.title("Enthalpy for values: a = %1.1f, kT = %1.2f" % (a, kT))
    plt.xlabel(r"$x_A$")
    plt.ylabel(r"$\Delta F(a,x_A)$")
    plt.savefig("figures/enthalpy_a%2.0f_kT%1.0f.pdf" % (a*10, kT*10))




def convergence_data_miner(a=1.3, kT=0.1):
    """
    Convergence test of this Monte Carlo program
    Runs through find_necessary_timesteps() for all xA.
    Creates a plot showing how necessary timesteps depends on xA
    """

    ### Max error
    max_error = error_benchmark(a, kT) # Benchmark error all other iterations must beat (have errors smaller than)
    # max_error = 0.7 # Average result from earlier runs with a=1.3 and kT=0.1
    print(max_error)

    ### Iteration settings
    Nxa = 98
    timestep_list = np.zeros(Nxa) 
    xA_list = np.linspace(0.01, 0.99, Nxa)
    
    ### Main program and timing
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
    np.save("timesteps" + astr, timestep_list) # Save for later.. 

    ### Load
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





####################
####### MAIN #######
####################

if __name__ == "__main__":
    ### Toggle comments to run the functions you want

    ### A few runs to create snapshots, quick < 1 min
    snapshot(a=0.9, kT=0.95)

    ### Many runs to create enthalpy as a function of xA, slow > 10 min
    # multirun(a=0.9, kT=0.95)

    ### Find necessary timesteps for different xA, extremely slow > 1 hour
    ### Needs some tweaking, doesn't always work right out of the box
    # convergence_data_miner()

    ### Table data
    # print(V_create(1.3))
    # print(V_create(0.9))
    
    plt.show()













