import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import njit
from functools import lru_cache

DEBUG = False


np.set_printoptions(linewidth=200)

np.random.seed(2)

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

    TOL = 0.5e-3

    # Flist = np.zeros(len(timesteps)) # Debugging
    Fprev = 0.0

    # n_outer = timesteps // 100 # Number of changes in T (Monte Carlo timesteps)
    # n_inner = timesteps // n_outer
    n_outer = 20
    n_inner = timesteps // n_outer
    print(n_outer)
    print(n_inner)

    Tlist = np.logspace(-1, 3, n_outer)

    # DEBUG
    # plt.figure()
    # plt.plot(np.arange(n_outer), Tlist, marker=".", linestyle="")

    n_recent = 10
    F_recent_list = np.arange(n_recent).astype(float)

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

                # k = (i*n_inner + j) % n_recent
                # F_recent_list[k] = F
                # F_avg = np.sum(F_recent_list) / n_recent
                # # print(F_recent_list, F_avg)
                # error = np.sqrt( np.sum( (F_avg - F_recent_list)**2 ) ) / np.abs(F_avg)
                # # print(error, "<", TOL,  F - Fprev)
                # if error < 10*TOL:
                    # print(error, "<", TOL,  F - Fprev)

                # if error < TOL:
                    # print("tolarance reached!")
                    # print(error, "<", TOL,  F - Fprev, i*n_inner + j)
                    # return grid, F
                Fprev = F

            else:
                grid = oldgrid
                # Swap back

    # DEBUG
    # plt.figure()
    # plt.plot(np.arange(timesteps), Flist, linestyle="", marker=".")

    return grid, F 


### Main program ###

@lru_cache(maxsize=32)
def enthalpy_start(a, kT):
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


    
def enthalpy_problem(timesteps_init, xAlist, a, kT):
    
    V, FA, FB = enthalpy_start(a, kT)

    N_xA = len(xAlist)
    Fs = np.zeros(N_xA)
    grids = np.zeros((N_xA, N*M))
    
    for i, xA in enumerate(xAlist):
        # timesteps = int(timesteps_init * 2*min(xA, (1-xA)))
        timesteps = timesteps_init
        startgrid = grid_create(xA)

        grids[i], Fs[i] = update_grid_find_F(V, kT, startgrid, timesteps)

        print("xA", xA, "F", Fs[i], "timesteps", timesteps)

    Fdeltas = Fs - FA*xAlist - FB*(1-xAlist)
    np.save("Fdeltas", Fdeltas)

    return Fdeltas, grids



def snapshot():
    timesteps = 4000

    # a = 1.3 # Aangstroem
    a = 0.9
    kT = 0.1 # 1/beta

    xAlist = np.array([0.25, 0.5, 0.75])

    start = time()
    _, grids = enthalpy_problem(timesteps, xAlist, a, kT)
    end = time()
    print("Snaptshot time", end-start)

    plt.figure()
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.matshow(grids[i].reshape((N, M)), fignum=False)


def multirun():
    timesteps = 2000

    a = 1.3 # Aangstroem
    # a = 0.9
    kT = 0.1 # 1/beta

    Nxa = 98
    xAlist = np.linspace(0.01, 0.99, Nxa)

    start = time()
    Fdeltas, _ = enthalpy_problem(timesteps, xAlist, a, kT)
    end = time()
    print("Multirun time", end-start)

    plt.figure()
    plt.plot(xAlist, Fdeltas)
    


def error_benchmark(a, kT):

    n_times = 10
    timelist = 10000 * np.ones(n_times, dtype=int)
    xA = np.array([0.5]) # Array because enthalpy_problem() requires a list
    Fdeltas = np.zeros(n_times) 

    for i, timesteps in enumerate(timelist):
        Fdeltas[i], _ = enthalpy_problem(timesteps, xA, a, kT)
        # print('Fdelta', Fdeltas[i])

    Fsum = np.sum(Fdeltas)
    Favg = Fsum / n_times
    rel_error_avg = np.sum( np.abs( Favg - Fdeltas) ) / Fsum

    return np.abs(rel_error_avg)



def error_finder(timesteps, xA, a, kT):

    n_times = 10
    Fdeltas = np.zeros(n_times)

    for i in range(n_times):
        Fdeltas[i], _ = enthalpy_problem(timesteps, np.array([xA]), a, kT)

    Fsum = np.sum(Fdeltas)
    Favg = Fsum / n_times
    # print("Favg", Favg)
    rel_error_avg = np.sum( np.abs( Favg - Fdeltas) ) / Fsum
    print("Relative error:", rel_error_avg)
    return np.abs(rel_error_avg)
        


def find_necessary_timesteps(xA, max_error, a, kT):
    timesteps = 100
    prevsteps = 0
    preverror = 0
    found_top_value = False
    while True:
        error = error_finder(timesteps, xA, a, kT)
        if error < max_error:
            preverror = error # Remove later
            found_top_value = True
            prevsteps = timesteps
            timesteps = int(timesteps * 0.9)
        elif found_top_value:
            print("Error < maxerror:", preverror, max_error) # Remove later
            break
        else:
            timesteps *= 2
    return prevsteps


def convergence_data_miner():
    a = 1.3 # Aangstroem
    # a = 0.9
    kT = 0.1 # 1/beta

    # max_error = error_benchmark(a, kT)
    # print(max_error)
    max_error = 0.008
    
    print(max_error)

    iterations = 10
    timestep_list = np.zeros(iterations) 
    xA_list = np.linspace(0.1, 0.9, iterations)
    for i, xA in enumerate(xA_list):
        timesteps = find_necessary_timesteps(0.2, max_error, a, kT)
        print("xA=%.2f needs %d timesteps" % (xA, timesteps))
        timestep_list[i] = timesteps
    
    np.save("timesteps", timestep_list)

    plt.figure()
    plt.plot(xA_list, timestep_list)






# def convergence_run():
    # a = 1.3 # Aangstroem
    # # a = 0.9
    # kT = 0.1 # 1/beta

    # n_times = 10
    # # timelist = (np.linspace(10, 100, n_times)**2).astype(int)
    # # timelist = 10000 + np.arange(n_times, dtype=int) * 1000
    # timelist = 10000 * np.ones(n_times, dtype=int)
    # # xAlist = np.array([0.25, 0.5, 0.75])
    # xAlist = np.array([0.5])
    # Fdeltas = np.zeros(n_times) 

    # for xA in xAlist:

        # for i, time in enumerate(timelist):
            # Fdeltas[i], _ = enthalpy_problem(time, np.array([xA]), a, kT)
            # # print('Fdelta', Fdeltas[i])

        # Fsum = np.sum(Fdeltas)
        # Favg = Fsum / n_times
        # rel_error_avg = np.sum( np.abs( Favg - Fdeltas) ) / Fsum



if __name__ == "__main__":

    # snapshot()
    # multirun()
    # convergence_run()
    convergence_data_miner()
    
    plt.show()













