import numpy as np
import matplotlib.pyplot as plt
from time import time

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
    xA = 0.5
    nA = int(xA * N*M)
    grid = np.ones(N*M, dtype=int) * 2
    grid[0:nA] = 1
    # np.random.shuffle(grid)
    return grid


# Create Hamiltonian
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
    swap1, swap2 = np.random.random_integers(0, L-1, size=2)
    if grid[swap1] == grid[swap2]:
        return swap(grid)
    return swap1, swap2



def F_calc(grid, V, kT):
    H = H_create(grid, V)
    E = np.linalg.eigvalsh(H)
    # F = 
    return -kT * np.sum( np.log(1 + np.exp(-E/kT) ) )




### Main program ###
def enthalpy_problem(timesteps):
    a = 1.3 # Aangstroem
    kT = 0.1 # 1/beta
    
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
    xA = 0.5
    grid = grid_create(xA)

    c = np.logspace(-1, 3, timesteps)
    rand_compare_W = np.random.rand(timesteps)

    Fprev = 0
    for i in range(timesteps):        

        swap1, swap2 = swap(grid)
        grid[swap1], grid[swap2] = grid[swap2], grid[swap1] 
        
        F = F_calc(grid, V, kT)
        W = np.exp( (F-Fprev) * c[i] )

        accept = (W < rand_compare_W[i])

        # Swap back if bigger
        if accept:
            Fprev = F
        else:
            # Swap back
            grid[swap1], grid[swap2] = grid[swap2], grid[swap1] 

    print()
    print("timesteps", timesteps)
    print("xA", xA)
    print("Enthalpy", F - FA*xA - FB*(1-xA))


if __name__ == "__main__":
    start = time()

    timesteps = 1000

    enthalpy_problem(timesteps)     

    end = time()
    print("time", end-start)
    













