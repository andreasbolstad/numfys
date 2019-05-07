import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import njit

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

# @njit
def V_create(a):
    VAA = epsAA * ( (gamAA/a)**6 - np.exp(-a/etaAA) )
    VAB = epsAB * ( (gamAB/a)**6 - np.exp(-a/etaAB) )
    VBB = epsBB * ( (gamBB/a)**6 - np.exp(-a/etaBB) )
    return [VAA, VAB, VBB]


# @njit
def grid_create(xA):
    # Matrix of A and B
    # A = 1, B = 2
    nA = int(xA * N*M)
    grid = np.ones(N*M, dtype=int) * 2
    grid[0:nA] = 1
    # np.random.shuffle(grid)
    return grid


# Create Hamiltonian
@njit
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


# def swap(grid):
    # swap1, swap2 = np.random.randint(L, size=2)
    # if grid[swap1] == grid[swap2]:
        # return swap(grid)
    # return swap1, swap2

def swap(grid):
    swap1 = 0
    swap2 = 0
    while grid[swap1] == grid[swap2]:
        swap1, swap2 = np.random.randint(L, size=2)
    grid[swap1], grid[swap2] = grid[swap2], grid[swap1] 
    return grid

# @njit
def F_calc(grid, V, kT):
    H = H_create(grid, V)
    E = np.linalg.eigvalsh(H)
    return -kT * np.sum( np.log(1 + np.exp(-E/kT) ) )


def update_grid_find_F(V, kT, grid, timesteps, multirun=True):

    Flist = np.zeros(timesteps) # Debugging
    # Fmin = 0.0
    Fprev = 0.0

    n_outer = timesteps // 100 # Number of changes in T (Monte Carlo timesteps)
    n_inner = timesteps // n_outer

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
                Flist[i*n_inner + j] = F # Debugging
                Fprev = F
                # if F < Fmin:
                    # Fmin = F
            else:
                grid = oldgrid
                # Swap back

    # mask = np.nonzero(Flist)
    if not multirun:
        plt.figure()
        plt.matshow(grid.reshape((N, M)))

        plt.figure()
        plt.plot(np.arange(timesteps), Flist, linestyle="", marker=".")

    return grid, F #Fmin


### Main program ###
def enthalpy_problem(timesteps_init, a, kT, multirun=False, load=False):
    
    V = V_create(a)

    # FA
    xA_A = 1
    grid = grid_create(xA_A)
    FA = F_calc(grid, V, kT)
    
    # FB
    xA_B = 0
    grid = grid_create(xA_B)
    FB = F_calc(grid, V, kT)

    # F(a) 
    N = 100

    ##############
    # Computations
    if multirun:
        F_a = np.zeros(N)
        xAs = np.linspace(0.1, 0.9, N)
    else:
        F_a = np.array([0])
        xAs = np.array([0.5])
    
    if not load: 
        for i, xA in enumerate(xAs):
            # timesteps = timesteps_init
            timesteps = int(timesteps_init * 2*min(xA, (1-xA)))
            grid = grid_create(xA)

            grid, F = update_grid_find_F(V, kT, grid, timesteps, multirun=multirun)

            print("xA", xA, "F", F, "timsteps", timesteps)
            F_a[i] = F

        Fdelta = F_a - FA*xAs - FB*(1-xAs)
        np.save("Fdeltas", Fdelta)

    ##########

    else:
        Fdelta = np.load("Fdeltas.npy")

    if multirun:
        plt.figure()
        plt.plot(xAs, Fdelta)
    
# def enthalpy_problem(timesteps_init, a, kT, multirun=False, load=False):
    
    # V = V_create(a)

    # # FA
    # xA_A = 1
    # grid = grid_create(xA_A)
    # FA = F_calc(grid, V, kT)
    
    # # FB
    # xA_B = 0
    # grid = grid_create(xA_B)
    # FB = F_calc(grid, V, kT)

    # # F(a) 
    # N = 100

    # ##############
    # # Computations
    # if multirun:
        # F_a = np.zeros(N)
        # xAs = np.linspace(0.1, 0.9, N)
    # else:
        # F_a = np.array([0])
        # xAs = np.array([0.5])
    
    # if not load: 
        # for i, xA in enumerate(xAs):
            # # timesteps = timesteps_init
            # timesteps = int(timesteps_init * 2*min(xA, (1-xA)))
            # grid = grid_create(xA)

            # grid, F = update_grid_find_F(V, kT, grid, timesteps, multirun=multirun)

            # print("xA", xA, "F", F, "timsteps", timesteps)
            # F_a[i] = F

        # Fdelta = F_a - FA*xAs - FB*(1-xAs)
        # np.save("Fdeltas", Fdelta)

    ##########

    else:
        Fdelta = np.load("Fdeltas.npy")

    if multirun:
        plt.figure()
        plt.plot(xAs, Fdelta)


if __name__ == "__main__":
    start = time()

    timesteps = 3000

    a = 1.3 # Aangstroem
    # a = 0.9
    kT = 0.1 # 1/beta

    enthalpy_problem(timesteps, a, kT, multirun=False, load=False)     

    end = time()
    print("time", end-start)
    
    plt.show()













