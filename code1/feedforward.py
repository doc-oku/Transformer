from linear import Linear
from relu import Relu


class FeedForward:
    def __init__(self, d_model):
        self.ds1 = Linear(4*d_model)
        self.ds2 = Linear(d_model)
        self.ac1 = Relu()

    def forward(self, x):
        x = self.ds1.forward(x)
        x = self.ac1.forward(x)
        x = self.ds2.forward(x)
        return x

    def backward(self, x):
        x = self.ds2.backward(x)
        x = self.ac1.backward(x)
        x = self.ds1.backward(x)
        return x

    def zero_gradient(self):
        self.ds1.zero_gradient()
        self.ds2.zero_gradient()

    def gradient(self):
        self.ds1.gradient()
        self.ds2.gradient()

    def adam(self, lr):
        self.ds1.adam(lr)
        self.ds2.adam(lr)

    def save(self, path):
        self.ds1.save(path+"ffds1")
        self.ds2.save(path+"ffds2")

    def load(self, path):
        self.ds1.load(path+"ffds1")
        self.ds2.load(path+"ffds2")
