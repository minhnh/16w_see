#!/usr/bin/env python3
import sys
import numpy as np


def remove_bad_cam(source, cam_name, dest):
    dt =[('timestamp', 'i8'), ('camname', 'S9'), ('x', 'f8'),
         ('y', 'f8'), ('theta', 'f8')]
    data = np.genfromtxt(source, dtype=dt)
    data = data[np.where(data['camname'] != cam_name)]
    #data = data['timestamp', 'camname', 'x', 'y', 'theta']
    #np.savetxt(dest, data)
    np.savetxt(dest, data, delimiter="\n", fmt="%d %s %f %f %f")
    pass


if __name__ == "__main__":
    source = sys.argv[1]
    cam_name = sys.argv[2]
    dest = sys.argv[3]
    remove_bad_cam(source, cam_name, dest)
    pass
