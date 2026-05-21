import numpy as np
from dropout import Dropout


class PositionEncoding:
    def __init__(self):
        self.dp = Dropout()

    def forward(self, x):
        token_size = x.shape[0]
        d_model = x.shape[1]
        self.scale = np.sqrt(d_model)

        pos = np.zeros(x.shape)
        p = np.arange(0, d_model)
        div_term = 10000**(p/d_model)

        for i in range(token_size):
            pos[i, 0::2] = np.sin(i/div_term[0::2])
            pos[i, 1::2] = np.cos(i/div_term[1::2])

        x = self.scale*x+pos
        x = self.dp.forward(x)
        return x

    def backward(self, x):
        x = self.scale*x
        x = self.dp.backward(x)
        return x
