import math
import matplotlib.pyplot as plt
import random

import numpy as np
import tensorflow as tf


def getData(N, variance):
    sigma = math.sqrt(variance)
    z = np.random.normal(0, sigma, N)

    x = np.random.random_sample(N)
    y = np.array([F(i) for i in x]) + z

    return x, y


def F(x):
    return math.cos(2 * math.pi * x)


x, y = getData(500, 0.015)
plt.scatter(x, y)
plt.show()

