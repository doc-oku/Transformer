import csv
import copy
import numpy as np


class Parameter:
    def __init__(self):
        self.initialize = True
        self.eps = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999

    def make(self):
        scale = np.sqrt(2.0)
        self.weight = np.random.normal(
            loc=0, scale=scale, size=(self.out_features, 1))

        self.d_weight = np.zeros(self.weight.shape)
        self.wm = np.zeros(self.weight.shape)
        self.wv = np.zeros(self.weight.shape)

    def forward(self, x):
        self.inputs_shape = x.shape

        if self.initialize:
            self.out_features = x.shape[2]
            self.make()
            self.initialize = False

        x = x.reshape(-1, x.shape[2])
        self.inputs = np.ones((x.shape[0], 1))

        self.scale = x.shape[0]
        x = np.dot(self.inputs, self.weight.T)
        x = x.reshape(self.inputs_shape)
        return x

    def backward(self, x):
        x = x.reshape(-1, x.shape[2])
        self.delta = copy.deepcopy(x)

    def zero_gradient(self):
        self.d_weight = 0.0

    def gradient(self):
        self.d_weight += np.dot(self.delta.T, self.inputs)

    def sgd(self, lr):
        self.weight -= lr * self.d_weight

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.999
        self.b1 *= beta1
        self.b2 *= beta2

        self.d_weight /= self.scale
        self.wm = beta1 * self.wm + (1.0 - beta1) * self.d_weight
        self.wv = beta2 * self.wv + (1.0 - beta2) * (self.d_weight ** 2)
        a = self.wm / (1.0 - self.b1)
        b = self.wv / (1.0 - self.b2)
        self.weight -= lr * a / (np.sqrt(b) + self.eps)

    def save(self, path):
        with open("parameter/"+path+"pm.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.weight)

    def load(self, path):
        self.path = path
