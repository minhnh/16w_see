import sys
import numpy as np
import matplotlib.pyplot as plt

f = sys.argv[1]
data = np.genfromtxt(f, usecols=(2,3,4))
#plt.scatter(data[:,0] / 1000., data[:,1] / 1000.)
#plt.xlim([0, 4])
#plt.ylim([0, 7])
#plt.show()

x_dir = np.cos(data[:,2] + np.pi)# + data[:,0]
y_dir = np.sin(data[:,2] + np.pi)# + data[:,1]
plt.quiver(data[:,0], data[:,1], x_dir, y_dir)
plt.show()
