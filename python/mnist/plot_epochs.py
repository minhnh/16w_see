#!/usr/bin/env python 3
import sys
import numpy as np
from matplotlib import pyplot as plt


def main(train_log_file):
    train_info = np.genfromtxt(train_log_file, delimiter=',')
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    acc_line, = plt.plot(train_info[:, 0], train_info[:, 1], '-r+',
                             markersize=4, label='train accuracy')
    val_line, = plt.plot(train_info[:, 0], train_info[:, 3], '-g+',
                             markersize=4, label='validation accuracy')
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(handles=[acc_line, val_line], loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fancybox=True,
              shadow=True, ncol=2)
    plt.title('Validation and training accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    ax.set_ylim([0.0, 1.1])
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    main(sys.argv[1])
