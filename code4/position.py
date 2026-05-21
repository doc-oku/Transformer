import numpy as np
from dropout import Dropout3d


class PositionEncoding:
    def __init__(self):
        self.dp = Dropout3d()

    def forward(self, x):
        batch_size = x.shape[0]
        token_size = x.shape[1]
        dimension = x.shape[2]
        self.scale = np.sqrt(dimension)

        pos = np.zeros(x.shape)
        p = np.arange(0, dimension)
        div_term = 10000**(p/dimension)

        for b in range(batch_size):
            for i in range(token_size):
                pos[b, i, 0::2] = np.sin(i/div_term[0::2])
                pos[b, i, 1::2] = np.cos(i/div_term[1::2])

        x = self.scale*x+pos
        x = self.dp.forward(x)
        return x

    def backward(self, x):
        x = self.scale*x
        x = self.dp.backward(x)
        return x

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass
