import os
import numpy as np
import matplotlib.pyplot as plt

data = []

for filename in os.listdir("./data"):
    if filename.endswith(".txt"):
        data.append(np.loadtxt("data/" + filename, skiprows=1))

nfiles = len(data)
print(nfiles, "number of files")

x = np.linspace(0, 1, len(data[0][:,0]))

# Major component probability (p_inf)
plt.figure()
plt.title(r"p_{\infty}")
for i in range(nfiles):
    y = data[i][:,0]
    plt.plot(x, y)

# <s>
plt.figure()
plt.xlim(0.4, 0.6)
plt.title("<s>")
for i in range(nfiles):
    y = data[i][:,1]
    plt.plot(x,y)

# chi
plt.figure()
plt.xlim(0.4, 0.6)
plt.title(r"\chi")
for i in range(nfiles):
    y = data[i][:,2]
    plt.plot(x,y) 
plt.show()

