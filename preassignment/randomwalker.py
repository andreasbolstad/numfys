from matplotlib import pyplot as plt
import randwalk
import numpy as np
from scipy.stats import linregress
from time import clock

if __name__ == "__main__":
    nw = 1e6
    max_t = 1000
    start = clock()
    a = randwalk.randwalk(nw, max_t)
    print(type(a))
    end = clock()
    a = np.log(a/nw)
    print(end-start)
    times = np.log(np.arange(1, max_t+1))
    alpha = -linregress(times, a)[0]
    print("alpha =", alpha)
    plt.figure()
    plt.plot(times, a)
    plt.show()

