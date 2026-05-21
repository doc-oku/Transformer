import numpy as np


class Gelu:
    def __init__(self):
        self.A = 0.044715
        self.B = np.sqrt(2.0/np.pi)

    def forward(self, x):
        self.inputs = x
        x1 = x**3
        x2 = self.A*x1
        x3 = x+x2
        x4 = self.B*x3
        self.x5 = np.tanh(x4)
        out = 0.5*x*(1+self.x5)
        return out

    def backward(self, x):
        x1 = 0.5*self.inputs*self.B*(1-self.x5**2)
        x2 = self.A*(3*self.inputs**2)*x1
        x3 = 0.5*(1+self.x5)
        back = (x1+x2+x3)*x
        return back
