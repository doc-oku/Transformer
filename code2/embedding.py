import csv
import numpy as np
import copy


class Embedding:
    def __init__(self, out_dimension):
        self.out_dimension = out_dimension
        self.eps = 1e-9
        self.b1 = 0.9
        self.b2 = 0.98
        self.initialize = True

    def make(self):
        scale = np.sqrt(2.0 / self.vocab_size)
        self.weight = np.random.normal(
            loc=0.0, scale=scale, size=(self.out_dimension, self.vocab_size))
        self.d_weight = np.zeros(self.weight.shape)
        self.wm = np.zeros(self.weight.shape)
        self.wv = np.zeros(self.weight.shape)

        try:
            with open("parameter/"+self.path+"emdw.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for j, row in enumerate(reader):
                    for i in range(len(row)):
                        self.weight[j, i] = row[i]
        except:
            print("error: embedding")

    def forward(self, x):
        self.token_size = x.shape[0]
        self.vocab_size = x.shape[1]

        if self.initialize:
            self.make()
            self.initialize = False

        self.inputs = copy.deepcopy(x)
        x = np.dot(self.inputs, self.weight.T)
        return x

    def backward(self, x):
        self.delta = copy.deepcopy(x)
        x = np.dot(self.delta, self.weight)
        return x

    def zero_gradient(self):
        self.d_weight = 0.0

    def gradient(self):
        self.d_weight += np.dot(self.delta.T, self.inputs)

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.98
        self.b1 *= beta1
        self.b2 *= beta2

        self.d_weight /= self.token_size
        self.wm = beta1 * self.wm + (1.0 - beta1) * self.d_weight
        self.wv = beta2 * self.wv + (1.0 - beta2) * (self.d_weight ** 2)
        m = self.wm / (1.0 - self.b1)
        v = self.wv / (1.0 - self.b2)
        self.weight -= lr * m / (np.sqrt(v) + self.eps)

    def save(self, path):
        with open("parameter/"+path+"emdw.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.weight)

    def load(self, path):
        self.path = path
