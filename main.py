import math
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf


def F(x):
    return 2 * math.pi * x


def f(x, parameters):
    d = len(parameters)

    y = 0
    for j in range(d):
        y += parameters[j] * math.pow(x, j)
    return y

def getData(N, variance):
    sigma = math.sqrt(variance)
    z = np.random.normal(0, sigma, N)

    x = np.random.random_sample(N)
    y = np.array([F(i) for i in x]) + z

    return x, y


def getMSE(x, y, parameters):
    d = len(parameters)
    N = len(x)

    sum = 0
    for i in range(N):
        result = 0
        for j in range(d):
            result = result + parameters[j] * math.pow(x[i], j)
        sum += math.pow(y[i] - result, 2)
    return sum / N


def fitData(x, y, initial_paramaters, learning_rate):
    d = len(initial_paramaters)
    N = len(x)
    epsilon = 0.000001

    theta_old = np.array(initial_paramaters)
    temp_theta = np.array(theta_old)
    count = 0
    while True:
        sum = 0
        sum2 = 0
        for i in range(N):
            sum += 2 * (y[i] - f(x[i], theta_old))
            sum2 += 2 * (y[i] - f(x[i], theta_old)) * x[i]
        temp_theta[0] += learning_rate * (sum / N)
        for j in range(1, d):
            temp_theta[j] += learning_rate * (sum2 / N)

        if abs(temp_theta[0] - theta_old[0]) < epsilon:
            break
        elif count > 10000:
            break
        theta_old = np.array(temp_theta)
        count += 1

    x_new, y_new = getData(N, 0.015)

    E_in = getMSE(x, y, theta_old)
    E_out = getMSE(x_new, y_new, theta_old)
    return theta_old, E_in, E_out


x, y = getData(500, 0.015)
print('x:', x)
print('y:', y)
print('getMSE:', getMSE(x, y, np.array([0.5, 0])))
optimal_params, E_in, E_out = fitData(x, y, np.array([0.2, 1.1, 0.4, 2.0]), 0.2)
print('getMSE:', getMSE(x, y, optimal_params))
print('E_in:', E_in)
print('E_out:', E_out)
reg_x = x
reg_y = [f(i, optimal_params) for i in reg_x]
print('optimal_params', optimal_params)
print('reg_x', reg_x[0])
print('reg_y', reg_y[0])
plt.scatter(x, y)
plt.scatter(reg_x, reg_y)
plt.show()

