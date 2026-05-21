import numpy as np
from dropout import Dropout3d
from parameter import Parameter


class PositionEncoding:
    def __init__(self):
        self.dp = Dropout3d()
        self.pm = Parameter()

    def forward(self, x):
        self.scale = np.sqrt(x.shape[2])
        pos = self.pm.forward(x)
        x = self.scale*x+pos
        x = self.dp.forward(x)
        return x

    def backward(self, x):
        self.pm.backward(x)
        x = self.scale*x
        x = self.dp.backward(x)
        return x

    def zero_gradient(self):
        self.pm.zero_gradient()

    def gradient(self):
        self.pm.gradient()

    def sgd(self, lr):
        self.pm.sgd(lr)

    def adam(self, lr):
        self.pm.adam(lr)
