import numpy as np


class SoftMax2d:
    def __init__(self):
        pass

    def forward(self, x):
        maximum = np.max(x, axis=1)
        maximum = maximum.reshape(-1, 1)
        exp_in = np.exp(x - maximum)
        sum1 = np.sum(exp_in, axis=1)
        sum1 = sum1.reshape(-1, 1)
        self.outputs = exp_in / sum1
        return self.outputs

    def backward(self, x):
        sum1 = np.sum(self.outputs * x, axis=1)
        sum1 = sum1.reshape(-1, 1)
        x = self.outputs * (x - sum1)
        return x

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass


class SoftMax3d:
    def __init__(self):
        pass

    def forward(self, x):
        self.inputs_shape = x.shape
        x = x.reshape(-1, x.shape[2])
        maximum = np.max(x, axis=1)
        maximum = maximum.reshape(-1, 1)
        exp_in = np.exp(x - maximum)
        sum1 = np.sum(exp_in, axis=1)
        sum1 = sum1.reshape(-1, 1)
        self.outputs = exp_in / sum1
        x = self.outputs.reshape(self.inputs_shape)
        return x

    def backward(self, x):
        x = x.reshape(-1, x.shape[2])
        sum1 = np.sum(self.outputs * x, axis=1)
        sum1 = sum1.reshape(-1, 1)
        x = self.outputs * (x - sum1)
        x = x.reshape(self.inputs_shape)
        return x

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass
