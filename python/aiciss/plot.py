#/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    f = sys.argv[1]
    data = np.genfromtxt(f, usecols=(2,3,4))
    #plt.scatter(data[:,0] / 1000., data[:,1] / 1000.)
    #plt.xlim([0, 4])
    #plt.ylim([0, 7])
    #plt.show()
    data_short = data#[500:610]#[np.where(data[:, 1] < 1800)]#[500:525]
    print(len(data))
    data_short[:, 2] = data_short[:, 2] + np.pi
    matrix = get_transform_matrix_from_pose(data_short[0])
    x_dir = np.cos(data_short[:,2])# + data[:,0]
    y_dir = np.sin(data_short[:,2])# + data[:,1]
    plt.quiver(data_short[:,0], data_short[:,1], x_dir, y_dir, color='red')
    plt.show()
    pass


if __name__ == "__main__":
    main()
    pass
