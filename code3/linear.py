import csv
import copy
import numpy as np


class Linear:
    def __init__(self, out_features):
        self.out_features = out_features
        self.initialize = True

    def make(self):
        self.eps = 1e-9
        self.b1 = 0.9
        self.b2 = 0.98

        scale = np.sqrt(2.0 / self.in_features)
        self.weight = np.random.normal(
            loc=0, scale=scale, size=(self.out_features, self.in_features))
        self.d_weight = np.zeros(self.weight.shape)
        self.wm = np.zeros(self.weight.shape)
        self.wv = np.zeros(self.weight.shape)
        self.bias = np.zeros((1, self.out_features))
        self.d_bias = np.zeros(self.bias.shape)
        self.bm = np.zeros(self.bias.shape)
        self.bv = np.zeros(self.bias.shape)

        try:
            with open("parameter/"+self.path+"dsw.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for j, row in enumerate(reader):
                    for i in range(len(row)):
                        self.weight[j, i] = float(row[i])

            with open("parameter/"+self.path+"dsb.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for j, row in enumerate(reader):
                    self.bias[0, j] = float(row[0])
        except:
            print("error: linear")

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.token_size = x.shape[1]

        if self.initialize:
            self.in_features = x.shape[2]
            self.make()
            self.initialize = False

        x = x.reshape(-1, x.shape[2])
        self.scale = x.shape[0]
        self.inputs = copy.deepcopy(x)
        x = np.dot(self.inputs, self.weight.T) + self.bias
        x = x.reshape(self.batch_size, self.token_size, -1)
        return x

    def backward(self, x):
        x = x.reshape(-1, x.shape[2])
        self.delta = copy.deepcopy(x)
        x = np.dot(self.delta, self.weight)
        x = x.reshape(self.batch_size, self.token_size, -1)
        return x

    def zero_gradient(self):
        self.d_weight = 0.0
        self.d_bias = 0.0

    def gradient(self):
        self.d_weight += np.dot(self.delta.T, self.inputs)
        self.d_bias += np.sum(self.delta, axis=0)

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.98

        self.b1 *= beta1
        self.b2 *= beta2

        self.d_bias /= self.scale
        self.bm = beta1 * self.bm + (1.0 - beta1) * self.d_bias
        self.bv = beta2 * self.bv + (1.0 - beta2) * (self.d_bias ** 2)
        a = self.bm / (1.0 - self.b1)
        b = self.bv / (1.0 - self.b2)
        self.bias -= lr * a / (np.sqrt(b) + self.eps)

        self.d_weight /= self.scale
        self.wm = beta1 * self.wm + (1.0 - beta1) * self.d_weight
        self.wv = beta2 * self.wv + (1.0 - beta2) * (self.d_weight ** 2)
        a = self.wm / (1.0 - self.b1)
        b = self.wv / (1.0 - self.b2)
        self.weight -= lr * a / (np.sqrt(b) + self.eps)

    def save(self, path):
        with open("parameter/"+path+"dsw.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.weight)

        bias = self.bias.reshape(-1, 1)
        with open("parameter/"+path+"dsb.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(bias)

    def load(self, path):
        self.path = path
