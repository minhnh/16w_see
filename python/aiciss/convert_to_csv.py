#!/usr/bin/env python3
import sys
import numpy as np
import os


def get_next_index(data, duration, start_index, time_step, index_estimate):
    start_time = data[start_index, 0]
    if data[-1, 0] < start_time + duration:
        return None

    if index_estimate > len(data):
        index_estimate = len(data) - 1
        pass

    time_change = data[index_estimate, 0] - start_time
    index_change = int((time_change - duration)/time_step + 0.5)

    if np.abs(index_change) < 1:
        if time_change > duration:
            return index_estimate
        else:
            while data[index_estimate, 0] - start_time < duration:
                index_estimate += 1
                pass
            return index_estimate
    else:
        return get_next_index(data, duration, start_index, time_step,
                              index_estimate - index_change)


def convert_to_csv(source_name, dest_name):
    data = np.genfromtxt(source_name, usecols=(0,2,3,4))
    timesteps = data[:, 0]
    timesteps = (timesteps - timesteps[0])/1000.
    data[:, 0] = timesteps
    data[:, 3] = data[:, 3] - np.pi
    data = data[::3]
    average_time_step = (data[-1, 0] - data[0, 0])/len(data)
    duration = 6.0
    split_num = int(len(data)/(duration/average_time_step))

    base_name = os.path.splitext(source_name)[0]
    # split a file into several trunks
    split_index = int(len(data)/split_num)
    #for i in range(split_num):
    estimate_index = split_index
    start_index = 0
    i = 0
    while estimate_index < len(data):
        file_name = base_name + "%02d.csv" % (i + 1)
        next_index = get_next_index(data, duration, start_index, average_time_step, estimate_index)
        #section = data[i*split_index:(i + 1)*split_index]
        #section[:, 1:] -= data[i*split_index, 1:]
        if next_index is None:
            print("reached None")
            break
        section = data[start_index:next_index + 1]
        np.savetxt(os.path.join(dest_name, file_name), section,  delimiter=',')
        #print("next_index: %d, start_index: %d, estimate_index: %d" % (next_index, start_index, estimate_index))
        #print("time change: %f" % (section[-1, 0] - section[0, 0]))
        # Update iteration
        start_index = next_index + 1
        estimate_index = next_index + split_index
        i += 1
        pass
    pass


if __name__ == "__main__":
    source_names = sys.argv[1:-1]
    dest_name = sys.argv[-1]
    for name in source_names:
        convert_to_csv(name, dest_name)
        pass
    pass
