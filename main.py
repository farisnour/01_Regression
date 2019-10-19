import math
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf


def F(x):
    return math.cos(2 * math.pi * x)


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
            result += parameters[j] * math.pow(x[i], j)
        if sum > 1e100 or result > 1e100 or result < -1e100:
            return 1e300
        sum += math.pow(y[i] - result, 2)
    return sum / N


def fitData(x, y, initial_paramaters, learning_rate):
    d = len(initial_paramaters)
    N = len(x)
    epsilon = 0.0000001

    theta_old = np.array(initial_paramaters)
    temp_theta = np.array(theta_old)
    loopCounter = 0
    while True:
        sum_vec = np.zeros(d)
        for i in range(N):
            x_vec = np.array([math.pow(x[i], j) for j in range(d)])
            sum_vec += (y[i] - f(x[i], theta_old)) * x_vec
        temp_theta += (learning_rate * 2 / (1.0 * N)) * sum_vec

        if abs(temp_theta[0] - theta_old[0]) < epsilon:
            print("optimal params found")
            break
        elif loopCounter > 10000:
            print("loop counter reached maximum, set a larger learning rate to get a better approximation")
            break
        theta_old = np.array(temp_theta)
        loopCounter += 1

    x_new, y_new = getData(N, 0.015)

    E_in = getMSE(x, y, theta_old)
    E_out = getMSE(x_new, y_new, theta_old)
    return theta_old, E_in, E_out


def experiment(N, d, noise_variance):
    M = 50
    optimal_params = np.zeros(shape=(M, d+1))
    E_in = np.zeros(M)
    E_out = np.zeros(M)
    for i in range(M):
        x, y = getData(N, noise_variance)
        optimal_params[i], E_in[i], E_out[i] = fitData(x, y, np.ones(d + 1), 0.5)
    E_in_avg = sum(E_in) / M
    E_out_avg = sum(E_out) / M

    vector = np.zeros(d + 1)
    for i in range(M):
        vector = vector + optimal_params[i]
    optimal_params_avg = vector / M

    x_new, y_new = getData(N, noise_variance)
    E_bias = getMSE(x_new, y_new, optimal_params_avg)

    x_final, y_final = getData(N, noise_variance)
    x_reg = sorted(x_final)
    y_reg = [f(i, optimal_params_avg) for i in x_reg]

    # plt.scatter(x_final, y_final)
    # plt.plot(x_reg, y_reg)
    # plt.show()

    return E_in_avg, E_out_avg, E_bias, x_final, y_final, optimal_params_avg


x, y = getData(50, 0.015)
print('Before - getMSE:', getMSE(x, y, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
optimal_params, E_in, E_out = fitData(x, y, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.5)
print('After  - getMSE:', getMSE(x, y, optimal_params))
print('optimal_params', optimal_params)
print('E_in:', E_in)
print('E_out:', E_out)
reg_x = sorted(x)
reg_y = [f(i, optimal_params) for i in reg_x]

plt.scatter(x, y)
plt.plot(reg_x, reg_y, 'r-')
plt.show()

# E_in_avg, E_out_avg, E_bias, x, y, optimal_params = experiment(2, 0, 0.01)
# print('E_in_avg:', E_in_avg)
# print('E_out_avg:', E_out_avg)
# print('E_bias:', E_bias)

# plt.scatter(x, y)
# x_reg = np.linspace(0, 1, 50)
# y_reg = [f(i, optimal_params) for i in x_reg]
# plt.plot(x_reg, y_reg)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# fig, axs = plt.subplots(2, 3)
#
# N_list = np.array([2, 5, 10, 20, 50, 100, 200])
# d_list = np.arange(6)
# sigma_list = np.array([0.01, 0.1, 1])
#
# for row in range(2):
#     for col in range(3):
#         i = 3 * row + col
#         E_in_avg, E_out_avg, E_bias, x, y, optimal_params = experiment(100, d_list[i], 0.01)
#         axs[row, col].scatter(x, y)
#         x_reg = sorted(x)
#         y_reg = [f(k, optimal_params) for k in x_reg]
#         axs[row, col].plot(x_reg, y_reg)
#         axs[row, col].set_title('N={}, d={}, variance={}'.format(100, d_list[i], 0.01))
#         plt.ylim(-1, 1)
#         plt.xlim(0, 1)
#
# plt.show()

# axs[0, 0].scatter(x, y)
# x_reg = sorted(x)
# y_reg = [f(i, optimal_params) for i in x_reg]
# axs[0, 0].plot(x_reg, y_reg)

# E_in_avg, E_out_avg, E_bias, x, y, optimal_params = experiment(5, 0, 0.01)
# axs[0, 1].scatter(x, y)
# x_reg = sorted(x)
# y_reg = [f(i, optimal_params) for i in x_reg]
# axs[0, 1].plot(x_reg, y_reg)


# plt.scatter(x, y)
# plt.plot(reg_x, reg_y)
# plt.show()

