import numpy as np
from embedding import Embedding
import attention


class MHSA:
    def __init__(self, mask, d_model):
        self.emd = Embedding(d_model)

        self.head = 6
        self.h_model = int(d_model/self.head)
        self.emdQ = [Embedding(self.h_model) for _ in range(self.head)]
        self.emdK = [Embedding(self.h_model) for _ in range(self.head)]
        self.emdV = [Embedding(self.h_model) for _ in range(self.head)]

        if mask == True:
            self.sa = [attention.SelfAttentionMask() for _ in range(self.head)]
        else:
            self.sa = [attention.SelfAttention() for _ in range(self.head)]

    def forward(self, x1, x2, x3):
        self.inputs_shape = x1.shape
        self.batch_size = x1.shape[0]
        self.token_size = x1.shape[1]

        x1 = x1.reshape(self.batch_size, self.token_size, self.head, -1)
        x2 = x2.reshape(self.batch_size, self.token_size, self.head, -1)
        x3 = x3.reshape(self.batch_size, self.token_size, self.head, -1)

        x1 = x1.transpose(2, 0, 1, 3)
        x2 = x2.transpose(2, 0, 1, 3)
        x3 = x3.transpose(2, 0, 1, 3)

        concat = np.zeros((self.head, self.batch_size,
                          self.token_size, self.h_model))

        for i in range(self.head):
            q = self.emdQ[i].forward(x1[i])
            k = self.emdK[i].forward(x2[i])
            v = self.emdV[i].forward(x3[i])
            concat[i] = self.sa[i].forward(q, k, v)

        concat = concat.transpose(1, 2, 0, 3)
        concat = concat.reshape(self.inputs_shape)
        x = self.emd.forward(concat)
        return x

    def backward(self, x):
        x = self.emd.backward(x)
        x = x.reshape(self.batch_size, self.token_size, self.head, -1)
        x = x.transpose(2, 0, 1, 3)

        q = np.zeros((self.head, self.batch_size,
                     self.token_size, self.h_model))
        k = np.zeros((self.head, self.batch_size,
                     self.token_size, self.h_model))
        v = np.zeros((self.head, self.batch_size,
                     self.token_size, self.h_model))

        for i in range(self.head):
            x1, x2, x3 = self.sa[i].backward(x[i])
            q[i] += self.emdQ[i].backward(x1)
            k[i] += self.emdK[i].backward(x2)
            v[i] += self.emdV[i].backward(x3)

        q = q.transpose(1, 2, 0, 3)
        q = q.reshape(self.inputs_shape)

        k = k.transpose(1, 2, 0, 3)
        k = k.reshape(self.inputs_shape)

        v = v.transpose(1, 2, 0, 3)
        v = v.reshape(self.inputs_shape)
        return q, k, v

    def zero_gradient(self):
        self.emd.zero_gradient()
        for i in range(self.head):
            self.emdQ[i].zero_gradient()
            self.emdK[i].zero_gradient()
            self.emdV[i].zero_gradient()

    def gradient(self):
        self.emd.gradient()
        for i in range(self.head):
            self.emdQ[i].gradient()
            self.emdK[i].gradient()
            self.emdV[i].gradient()

    def adam(self, lr):
        self.emd.adam(lr)
        for i in range(self.head):
            self.emdQ[i].adam(lr)
            self.emdK[i].adam(lr)
            self.emdV[i].adam(lr)

    def save(self, path):
        self.emd.save(path+"mhsaemd")
        for i in range(self.head):
            self.emdQ[i].save(path+"mhsaemdq"+str(i))
            self.emdK[i].save(path+"mhsaemdk"+str(i))
            self.emdV[i].save(path+"mhsaemdv"+str(i))

    def load(self, path):
        self.emd.load(path+"mhsaemd")
        for i in range(self.head):
            self.emdQ[i].load(path+"mhsaemdq"+str(i))
            self.emdK[i].load(path+"mhsaemdk"+str(i))
            self.emdV[i].load(path+"mhsaemdv"+str(i))
