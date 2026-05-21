import csv
import numpy as np


class LayerNorm:
    def __init__(self):
        self.eps = 1e-9
        self.b1 = 0.9
        self.b2 = 0.98
        self.initialize = True

    def make(self):
        self.gamma = 1.0
        self.beta = 0.0
        self.d_gamma = 0
        self.d_beta = 0
        self.gm = 0
        self.gv = 0
        self.bm = 0
        self.bv = 0

        try:
            with open("parameter/" + self.path + "nmg.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.gamma = float(row[0])

            with open("parameter/" + self.path + "nmb.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.beta = float(row[0])
        except:
            print("error: norm")

    def forward(self, x):
        self.token_size = x.shape[0]
        self.d_model = x.shape[1]

        if self.initialize:
            self.make()
            self.initialize = False

        ave = np.sum(x, axis=1)
        ave /= self.d_model
        ave = ave.reshape(-1, 1)

        self.d_input = x - ave
        x = np.square(self.d_input)
        var = np.sum(x, axis=1)
        var /= self.d_model
        var = var.reshape(-1, 1)

        self.buf1 = np.sqrt(var + self.eps)
        self.buf3 = self.buf1 ** 3
        self.xb = self.d_input / self.buf1
        x = self.gamma * self.xb + self.beta
        x = x.reshape(self.token_size, -1)
        return x

    def backward(self, x):
        self.delta = x
        d_xb = self.gamma * self.delta
        d_var = np.sum(self.d_input * d_xb, axis=1)
        d_var = d_var.reshape(-1, 1)
        d_var *= -0.5 / self.buf3

        sum1 = np.sum(d_xb, axis=1)
        sum1 = sum1.reshape(-1, 1)
        sum1 /= -self.buf1

        sum2 = np.sum(self.d_input, axis=1)
        sum2 = sum2.reshape(-1, 1)
        sum2 *= -2.0 * d_var / self.d_model
        d_ave = sum1 + sum2

        x = d_xb / self.buf1
        x += (2.0 * self.d_input * d_var + d_ave) / self.d_model
        x = x.reshape(self.token_size, -1)
        return x

    def zero_gradient(self):
        self.d_gamma = 0.0
        self.d_beta = 0.0

    def gradient(self):
        self.d_gamma += np.sum(self.delta * self.xb)
        self.d_beta += np.sum(self.delta)

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.98
        self.b1 *= beta1
        self.b2 *= beta2

        self.d_gamma /= self.token_size
        self.gm = beta1 * self.gm + (1.0 - beta1) * self.d_gamma
        self.gv = beta2 * self.gv + (1.0 - beta2) * (self.d_gamma ** 2)
        m = self.gm / (1.0 - self.b1)
        v = self.gv / (1.0 - self.b2)
        self.gamma -= lr * m / (np.sqrt(v) + self.eps)

        self.d_beta /= self.token_size
        self.bm = beta1 * self.bm + (1.0 - beta1) * self.d_beta
        self.bv = beta2 * self.bv + (1.0 - beta2) * (self.d_beta ** 2)
        m = self.bm / (1.0 - self.b1)
        v = self.bv / (1.0 - self.b2)
        self.beta -= lr * m / (np.sqrt(v) + self.eps)

    def save(self, path):
        with open("parameter/" + path + "nmg.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.gamma])

        with open("parameter/" + path + "nmb.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.beta])

    def load(self, path):
        self.path = path
