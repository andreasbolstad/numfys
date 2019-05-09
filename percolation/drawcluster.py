import numpy as np
import matplotlib.pyplot as plt

filename = 'clusters.txt'
f = open(filename, 'r')
ns, l, n, m = np.fromstring(f.readline(), dtype=int, sep="\t")
pcts = np.fromstring(f.readline(), sep="\t")
clusters = np.loadtxt(filename, skiprows=2)

nfiles = len(clusters) 
clusters = np.reshape(clusters, (nfiles, l, l))

fig, axarr = plt.subplots(2,3)

for i, c in enumerate(clusters):
    axarr[i//3, i%3].matshow(c)

plt.savefig("cluster_drawing.pdf")

plt.show()
