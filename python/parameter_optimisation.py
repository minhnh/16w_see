#/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize

def dF1(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[0] * v**2 + alphas[1] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (v**2 / k - ((v - v_hat) * v**2) / (k**2))
    return d

def dF2(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[0] * v**2 + alphas[1] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (omega**2 / k - ((v - v_hat) * omega**2) / (k**2))
    return d

def dF3(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[2] * v**2 + alphas[3] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (v**2 / k - ((omega - omega_hat) * v**2) / (k**2))
    return d

def dF4(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[2] * v**2 + alphas[3] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (omega**2 / k - ((omega - omega_hat) * omega**2) / (k**2))
    return d

def dF5(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[4] * v**2 + alphas[5] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (v**2 / k - (gamma_hat * v**2) / (k**2))
    return d

def dF6(alphas, v, omega, v_hat, omega_hat, gamma_hat):
    k = (alphas[4] * v**2 + alphas[5] * omega**2)
    if k < 1e-100:
        k = 1e-100
    d = -0.5 * (omega**2 / k - (gamma_hat * omega**2) / (k**2))
    return d

def F(alphas, X):
    s = 0.
    for x in X:
        k = (alphas[0] * x[0]**2 + alphas[1] * x[1]**2)
        if k < 1e-100:
            k = 1e-100
        temp = -0.5 * (np.log(2 * np.pi) + np.log(k) + (x[0] - x[2]) / k)            
        
        k = (alphas[2] * x[0]**2 + alphas[3] * x[1]**2)
        if k < 1e-100:
            k = 1e-100
        temp += -0.5 * (np.log(2 * np.pi) + np.log(k) + (x[1] - x[3]) / k)

        k = (alphas[4] * x[0]**2 + alphas[5] * x[1]**2)
        if k < 1e-100:
            k = 1e-100
        temp += -0.5 * (np.log(2 * np.pi) + np.log(k) + x[4] / k)

        s += temp
    return -1. * s

def F_prime(alphas, X):
    d = np.zeros(len(alphas))
    for i in range(len(alphas)):
        s = 0.
        if i == 0:
            for x in X:
                s += dF1(alphas, *x)
        elif i == 1:
            for x in X:
                s += dF2(alphas, *x)
        elif i == 2:
            for x in X:
                s += dF3(alphas, *x)
        elif i == 3:
            for x in X:
                s += dF4(alphas, *x)
        elif i == 4:
            for x in X:
                s += dF5(alphas, *x)
        elif i == 5:
            for x in X:
                s += dF6(alphas, *x)
        d[i] = s
    return -1. * d

def optimise_parameters(data, init_alphas):
    bounds = [(0., None), (0., None), (0., None), (0., None), (0., None), (0., None)]
    r = minimize(F, init_alphas, args=(data,), method='L-BFGS-B', jac=F_prime, bounds=bounds, options={'disp': True})
    alphas = r.x
    return alphas