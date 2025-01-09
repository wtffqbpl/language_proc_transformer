#! coding: utf-8


import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    return 1 / math.sqrt(2 * math.pi * sigma ** 2) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

x = np.arange(-7, 7, 0.01)

params = [ (0, 1), (0, 2), (3, 1), ]

for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f'mu={mu}, sigma={sigma}')

plt.legend()
plt.show()
