import numpy as np
import copy
from softmax import SoftMax2d
from matmul import MatMul
from dropout import Dropout2d


class SelfAttentionMask:
    def __init__(self):
        self.sm = SoftMax2d()
        self.mm1 = MatMul()
        self.mm2 = MatMul()
        self.dp = Dropout2d()
        self.minimum = np.iinfo(np.int64).min

    def forward(self, q, k, v):
        self.inputs_shape = q.shape
        self.batch_size = q.shape[0]
        self.token_size = q.shape[1]
        self.scale = 1/np.sqrt(q.shape[2])

        mask = np.zeros((self.token_size, self.token_size))
        for i in range(self.token_size):
            for j in range(i+1, self.token_size):
                mask[i, j] = self.minimum

        attention = np.zeros(self.inputs_shape)

        for b in range(self.batch_size):
            qk = self.mm1.forward(q[b], k[b].T)
            qk *= self.scale
            qk += mask
            x = self.sm.forward(qk)
            x = self.dp.forward(x)
            attention[b] = self.mm2.forward(x, v[b])
        return attention

    def backward(self, x):
        delta = copy.deepcopy(x)
        
        mask = np.ones((self.token_size, self.token_size))
        for i in range(self.token_size):
            for j in range(i+1, self.token_size):
                mask[i, j] = 0.0

        q = np.zeros(self.inputs_shape)
        k = np.zeros(self.inputs_shape)
        k = k.transpose(0, 2, 1)
        v = np.zeros(self.inputs_shape)

        for b in range(self.batch_size):
            x, v[b] = self.mm2.backward(delta[b])
            x = self.dp.backward(x)
            qk = self.sm.backward(x)
            qk *= self.scale
            qk *= mask
            q[b], k[b] = self.mm1.backward(qk)

        kT = k.transpose(0, 2, 1)
        return q, kT, v

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass


class SelfAttention:
    def __init__(self):
        self.sm = SoftMax2d()
        self.mm1 = MatMul()
        self.mm2 = MatMul()
        self.dp = Dropout2d()

    def forward(self, q, k, v):
        self.inputs_shape = q.shape
        self.batch_size = q.shape[0]
        self.scale = 1/np.sqrt(q.shape[2])

        attention = np.zeros(self.inputs_shape)
        for b in range(self.batch_size):
            qk = self.mm1.forward(q[b], k[b].T)
            qk *= self.scale
            x = self.sm.forward(qk)
            x = self.dp.forward(x)
            attention[b] = self.mm2.forward(x, v[b])
        return attention

    def backward(self, x):
        delta = copy.deepcopy(x)
        q = np.zeros(self.inputs_shape)
        k = np.zeros(self.inputs_shape)
        k = k.transpose(0, 2, 1)
        v = np.zeros(self.inputs_shape)

        for b in range(self.batch_size):
            x, v[b] = self.mm2.backward(delta[b])
            x = self.dp.backward(x)
            qk = self.sm.backward(x)
            qk *= self.scale
            q[b], k[b] = self.mm1.backward(qk)

        kT = k.transpose(0, 2, 1)
        return q, kT, v

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass
