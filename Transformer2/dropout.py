import numpy as np


class Dropout:
    def __init__(self):
        pass

    def forward(self, x):
        self.drop = np.random.rand(x.shape[0], x.shape[1])
        x = np.where(self.drop > 0.1, x, 0)
        return x

    def backward(self, x):
        x = np.where(self.drop > 0.1, x, 0)
        return x
