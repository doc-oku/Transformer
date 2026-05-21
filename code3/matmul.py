import numpy as np


class MatMul:
    def __init__(self):
        pass

    def forward(self, x1, x2):
        self.x1T = x1.T
        self.x2T = x2.T
        x = np.dot(x1, x2)
        return x

    def backward(self, x):
        x1 = np.dot(x, self.x2T)
        x2 = np.dot(self.x1T, x)
        return x1, x2
