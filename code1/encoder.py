from embedding import Embedding
from mhsa import MHSA
from norm import LayerNorm
from feedforward import FeedForward
from dropout import Dropout
import copy


class Encoder:
    def __init__(self, d_model):
        self.emdQ = Embedding(d_model)
        self.emdK = Embedding(d_model)
        self.emdV = Embedding(d_model)
        self.ms1 = MHSA(False, d_model)
        self.nm1 = LayerNorm()
        self.nm2 = LayerNorm()
        self.ff = FeedForward(d_model)
        self.dp1 = Dropout()
        self.dp2 = Dropout()

    def forward(self, x):
        shortcut = copy.deepcopy(x)
        q = self.emdQ.forward(x)
        k = self.emdK.forward(x)
        v = self.emdV.forward(x)
        x = self.ms1.forward(q, k, v)
        x = self.dp1.forward(x)
        x += shortcut
        x = self.nm1.forward(x)
        shortcut = copy.deepcopy(x)
        x = self.ff.forward(x)
        x = self.dp2.forward(x)
        x += shortcut
        x = self.nm2.forward(x)
        return x

    def backward(self, x):
        x = self.nm2.backward(x)
        shortcut = copy.deepcopy(x)
        x = self.dp2.backward(x)
        x = self.ff.backward(x)
        x += shortcut
        x = self.nm1.backward(x)
        shortcut = copy.deepcopy(x)
        x = self.dp1.backward(x)
        q, k, v = self.ms1.backward(x)
        q = self.emdQ.backward(q)
        k = self.emdK.backward(k)
        v = self.emdV.backward(v)
        x = q+k+v+shortcut
        return x

    def zero_gradient(self):
        self.ff.zero_gradient()
        self.ms1.zero_gradient()
        self.nm1.zero_gradient()
        self.nm2.zero_gradient()
        self.emdQ.zero_gradient()
        self.emdK.zero_gradient()
        self.emdV.zero_gradient()

    def gradient(self):
        self.ff.gradient()
        self.ms1.gradient()
        self.nm1.gradient()
        self.nm2.gradient()
        self.emdQ.gradient()
        self.emdK.gradient()
        self.emdV.gradient()

    def adam(self, lr):
        self.ff.adam(lr)
        self.ms1.adam(lr)
        self.nm1.adam(lr)
        self.nm2.adam(lr)
        self.emdQ.adam(lr)
        self.emdK.adam(lr)
        self.emdV.adam(lr)

    def save(self, path):
        self.ff.save(path+"encff")
        self.emdQ.save(path+"encemdq")
        self.emdK.save(path+"encemdk")
        self.emdV.save(path+"encemdv")
        self.nm1.save(path+"encnm1")
        self.nm2.save(path+"encnm2")
        self.ms1.save(path+"encms1")

    def load(self, path):
        self.ff.load(path+"encff")
        self.emdQ.load(path+"encemdq")
        self.emdK.load(path+"encemdk")
        self.emdV.load(path+"encemdv")
        self.nm1.load(path+"encnm1")
        self.nm2.load(path+"encnm2")
        self.ms1.load(path+"encms1")
