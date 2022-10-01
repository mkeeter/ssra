import pylab as plt
import numpy as np

data = open('data.txt').read()
data = [[int(i) for i in line.split(' ') if i] for line in data.split('\n') if line]
data = np.array(data)

fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
ax.plot(data[:,0], data[:,1] * 2)
ax.set_xlabel("Register count")
ax.set_ylabel("Memory operations")
plt.savefig("pressure.svg", bbox_inches='tight', transparent=True)
