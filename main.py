import math
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf


def F(x):
    return math.cos(2 * math.pi * x)


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


x, y = getData(500, 0.015)
print('x:', x)
print('y:', y)
print('getMSE:', getMSE(x, y, np.array([1, 3, 2, 0.3])))
# plt.scatter(x, y)
# plt.show()

