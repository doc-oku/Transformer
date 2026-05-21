import numpy as np
from softmax import SoftMax
from matmul import MatMul
from dropout import Dropout


class SelfAttentionMask:
    def __init__(self):
        self.sm = SoftMax()
        self.mm1 = MatMul()
        self.mm2 = MatMul()
        self.dp = Dropout()
        self.minimum = np.iinfo(np.int64).min

    def forward(self, q, k, v):
        self.token_size = q.shape[0]
        self.scale = 1/np.sqrt(q.shape[1])
        mask = np.zeros((self.token_size, self.token_size))

        for i in range(self.token_size):
            for j in range(i+1, self.token_size):
                mask[i, j] = self.minimum

        k = k.T
        qk = self.mm1.forward(q, k)
        qk *= self.scale
        qk += mask
        x = self.sm.forward(qk)
        x = self.dp.forward(x)
        x = self.mm2.forward(x, v)
        return x

    def backward(self, x):
        mask = np.ones((self.token_size, self.token_size))

        for i in range(self.token_size):
            for j in range(i+1, self.token_size):
                mask[i, j] = 0.0

        x, v = self.mm2.backward(x)
        x = self.dp.backward(x)
        qk = self.sm.backward(x)
        qk *= self.scale
        qk *= mask
        q, k = self.mm1.backward(qk)
        k = k.T
        return q, k, v

class SelfAttention:
    def __init__(self):
        self.sm = SoftMax()
        self.mm1 = MatMul()
        self.mm2 = MatMul()
        self.dp = Dropout()

    def forward(self, q, k, v):
        self.token_size = q.shape[0]
        self.scale = 1/np.sqrt(q.shape[1])
        qk = self.mm1.forward(q, k.T)
        qk *= self.scale
        x = self.sm.forward(qk)
        x = self.dp.forward(x)
        x = self.mm2.forward(x, v)
        return x

    def backward(self, x):
        x, v = self.mm2.backward(x)
        x = self.dp.backward(x)
        qk = self.sm.backward(x)
        qk *= self.scale
        q, k = self.mm1.backward(qk)
        return q, k.T, v
