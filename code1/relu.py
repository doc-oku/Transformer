import numpy as np


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        self.grad = np.where(x > 0.0, 1.0, 0.0)
        out = np.where(x > 0.0, x, 0.0)
        return out

    def backward(self, x):
        back = x * self.grad
        return back
