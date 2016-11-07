#!/usr/bin/env python3
import sys
import numpy as np
import os


def convert_to_csv(source_name, dest_name):
    data = np.genfromtxt(source_name, usecols=(0,2,3,4))
    timesteps = data[:, 0]
    timesteps = (timesteps - timesteps[0])/1000.
    data[:, 0] = timesteps
    data[:, 3] = data[:, 3] - np.pi
    data = data[::3]
    split_num = int(len(data)/(6.0/((data[-1, 0] - data[0, 0])/len(data))))

    base_name = os.path.splitext(source_name)[0]
    # split a file into several trunks
    split_index = int(len(data)/split_num)
    for i in range(split_num):
        file_name = base_name + str(i + 1) + ".csv"
        np.savetxt(os.path.join(dest_name, file_name), data[i*split_index:(i + 1)*split_index],  delimiter=',')
        pass
    pass


if __name__ == "__main__":
    source_name = sys.argv[1]
    dest_name = sys.argv[2]
    convert_to_csv(source_name, dest_name)
    pass
