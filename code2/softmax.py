import numpy as np


class SoftMax:
    def __init__(self):
        pass

    def forward(self, x):
        maximum = np.max(x, axis=1)
        maximum = maximum.reshape(-1, 1)
        exp_in = np.exp(x - maximum)
        sum1 = np.sum(exp_in, axis=1)
        sum1 = sum1.reshape(-1, 1)
        self.out = exp_in / sum1
        return self.out

    def backward(self, x):
        sum1 = np.sum(self.out * x, axis=1)
        sum1 = sum1.reshape(-1, 1)
        x = self.out * (x - sum1)
        return x
