import numpy as np


class Dropout2d:
    def __init__(self):
        pass

    def forward(self, x):
        self.drop = np.random.rand(x.shape[0], x.shape[1])
        x = np.where(self.drop > 0.1, x, 0)
        return x

    def backward(self, x):
        x = np.where(self.drop > 0.1, x, 0)
        return x

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass


class Dropout3d:
    def __init__(self):
        pass

    def forward(self, x):
        self.drop = np.random.rand(x.shape[0], x.shape[1], x.shape[2])
        x = np.where(self.drop > 0.1, x, 0)
        return x

    def backward(self, x):
        x = np.where(self.drop > 0.1, x, 0)
        return x

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass
