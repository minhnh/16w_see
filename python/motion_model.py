import glob
import os
import numpy as np
from parameter_optimisation import optimise_parameters

class Data(object):
    def __init__(self):
        self.trajectories = list()
        self.trajectories_end = list()
        self.time_deltas = list()

def read_data(directory_names):
    data_store = Data()
    for directory_name in directory_names:
        os.chdir(directory_name)
        trajectories = list()
        trajectories_end = list()
        times = list()
        
        files = list()
        for f in glob.glob("*.csv"):
            files.append(f)
    
        for f in files:
            data = np.loadtxt(f,delimiter=',')
            #data: shape(samples,4) -- [[time_stamp,x,y,gamma]]           
            
            poses = np.zeros((data.shape[0], 3))
            time_deltas = np.zeros(data.shape[0]-1)
            
            poses[0,0] = 0.
            poses[0,1] = 0.
            poses[0,2] = 0.
            for i in xrange(1, data.shape[0]):
                poses[i,0] = data[i,1] - data[0,1]
                poses[i,1] = data[i,2] - data[0,2]
                poses[i,2] = data[i,3] - data[0,3]
                #time_deltas[i-1] = data[i,0] - data[i-1,0]
                time_deltas[i-1] = data[i,0] - data[0,0]
    
            trajectories.append(poses)
            trajectories_end.append(poses[-1])
            times.append(time_deltas)
    
        data_store.trajectories.append(trajectories)
        data_store.trajectories_end.append(trajectories_end)
        data_store.time_deltas.append(times)

        os.chdir('..')
    return data_store

def motion_model_velocity(xt, ut, x_prevt, delta_t):
    mu = 0.5 * ((x_prevt[0] - xt[0]) * np.cos(x_prevt[2]) + (x_prevt[1] - xt[1]) * np.sin(x_prevt[2])) / ((x_prevt[1] - xt[1]) *
    np.cos(x_prevt[2]) - (x_prevt[0] - xt[0]) * np.sin(x_prevt[2]))
    x_star = 0.5 * (x_prevt[0] + xt[0]) + mu * (x_prevt[1] - xt[1])
    y_star = 0.5 * (x_prevt[1] + xt[1]) + mu * (xt[0] - x_prevt[0])
    r_star = np.sqrt((x_prevt[0] - x_star)**2 + (x_prevt[1] - y_star)**2)
    delta_theta = np.arctan2(xt[1] - y_star, xt[0] - x_star) - np.arctan2(x_prevt[1] - y_star, x_prevt[0] - x_star)
    v_hat = delta_theta / delta_t * r_star
    omega_hat = delta_theta / delta_t
    gamma_hat = (xt[2] - x_prevt[2]) / delta_t - omega_hat
    return np.array([ut[0], ut[1], v_hat, omega_hat, gamma_hat])


#robot run time: 6.28s, 3.93s, 6.28s, 3.93s, 6.28s
directory_names = ['straight', 'left', 'slight_left', 'right', 'slight_right']
v = [-10., -10., -10., -10., -10.]
omega = [0., np.radians(-22.92), np.radians(-4.77), np.radians(22.92), np.radians(4.77)]
data = read_data(directory_names)

motion_model_data = list()
for i, trajectories in enumerate(data.trajectories):
    for j, trajectory in enumerate(trajectories):
        for k in xrange(len(trajectory)-1):
            delta_t = data.time_deltas[i][j][k]
            #mmv = motion_model_velocity(trajectory[k+1], (v[i], omega[i]), trajectory[k], delta_t)
            mmv = motion_model_velocity(trajectory[k+1], (v[i], omega[i]), trajectory[0], delta_t)
            motion_model_data.append(mmv)
motion_model_data = np.array(motion_model_data)


init_alphas = np.random.uniform(1e-7, 1e-5, 6)
alphas = optimise_parameters(motion_model_data, init_alphas)
print alphas