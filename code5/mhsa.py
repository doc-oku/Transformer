import numpy as np
from embedding import Embedding
from attention import SelfAttentionMask
from attention import SelfAttention


class MHSA:
    def __init__(self, mask, d_model):
        self.head = 6
        self.emd = Embedding(d_model)
        self.h_model = int(d_model/self.head)
        self.emdQ = Embedding(d_model)
        self.emdK = Embedding(d_model)
        self.emdV = Embedding(d_model)

        if mask == True:
            self.sa = [SelfAttentionMask() for _ in range(self.head)]
        else:
            self.sa = [SelfAttention() for _ in range(self.head)]

    def forward(self, x1, x2, x3):
        self.inputs_shape = x1.shape
        self.batch_size = x1.shape[0]
        self.token_size = x1.shape[1]

        q = self.emdQ.forward(x1)
        k = self.emdK.forward(x2)
        v = self.emdV.forward(x3)

        q = q.reshape(self.batch_size, self.token_size, self.head, -1)
        k = k.reshape(self.batch_size, self.token_size, self.head, -1)
        v = v.reshape(self.batch_size, self.token_size, self.head, -1)

        q = q.transpose(2, 0, 1, 3)
        k = k.transpose(2, 0, 1, 3)
        v = v.transpose(2, 0, 1, 3)

        concat = np.zeros((self.head, self.batch_size,
                          self.token_size, self.h_model))

        for i in range(self.head):
            concat[i] = self.sa[i].forward(q[i], k[i], v[i])

        concat = concat.transpose(1, 2, 0, 3)
        concat = concat.reshape(self.inputs_shape)
        x = self.emd.forward(concat)
        return x

    def backward(self, x):
        x = self.emd.backward(x)
        x = x.reshape(self.batch_size, self.token_size, self.head, -1)
        x = x.transpose(2, 0, 1, 3)

        x1 = np.zeros((self.head, self.batch_size,
                      self.token_size, self.h_model))
        x2 = np.zeros((self.head, self.batch_size,
                      self.token_size, self.h_model))
        x3 = np.zeros((self.head, self.batch_size,
                      self.token_size, self.h_model))

        for i in range(self.head):
            x1[i], x2[i], x3[i] = self.sa[i].backward(x[i])

        x1 = x1.transpose(1, 2, 0, 3)
        x1 = x1.reshape(self.inputs_shape)

        x2 = x2.transpose(1, 2, 0, 3)
        x2 = x2.reshape(self.inputs_shape)

        x3 = x3.transpose(1, 2, 0, 3)
        x3 = x3.reshape(self.inputs_shape)

        q = self.emdQ.backward(x1)
        k = self.emdK.backward(x2)
        v = self.emdV.backward(x3)
        return q, k, v

    def zero_gradient(self):
        self.emd.zero_gradient()
        self.emdQ.zero_gradient()
        self.emdK.zero_gradient()
        self.emdV.zero_gradient()

    def gradient(self):
        self.emd.gradient()
        self.emdQ.gradient()
        self.emdK.gradient()
        self.emdV.gradient()

    def adam(self, lr):
        self.emd.adam(lr)
        self.emdQ.adam(lr)
        self.emdK.adam(lr)
        self.emdV.adam(lr)

    def save(self, path):
        self.emd.save(path+"mhsaemd")
        self.emdQ.save(path+"mhsaemdq")
        self.emdK.save(path+"mhsaemdk")
        self.emdV.save(path+"mhsaemdv")

    def load(self, path):
        self.emd.load(path+"mhsaemd")
        self.emdQ.load(path+"mhsaemdq")
        self.emdK.load(path+"mhsaemdk")
        self.emdV.load(path+"mhsaemdv")
