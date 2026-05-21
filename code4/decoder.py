from embedding import Embedding
from mhsa import MHSA
from norm import LayerNorm
from feedforward import FeedForward
from dropout import Dropout3d
import copy


class Decoder:
    def __init__(self, d_model):
        self.emdQ = Embedding(d_model)
        self.emdK = Embedding(d_model)
        self.emdV = Embedding(d_model)
        self.ms1 = MHSA(True, d_model)
        self.ms2 = MHSA(False, d_model)
        self.nm1 = LayerNorm()
        self.nm2 = LayerNorm()
        self.nm3 = LayerNorm()
        self.ff = FeedForward(d_model)
        self.dp1 = Dropout3d()
        self.dp2 = Dropout3d()
        self.dp3 = Dropout3d()

    def forward(self, x1, x2):
        shortcut = copy.deepcopy(x1)
        q = self.emdQ.forward(x1)
        k = self.emdK.forward(x1)
        v = self.emdV.forward(x1)
        x = self.ms1.forward(q, k, v)
        x = self.dp1.forward(x)
        x += shortcut
        x = self.nm1.forward(x)
        shortcut = copy.deepcopy(x)
        x = self.ms2.forward(x, x2, x2)
        x = self.dp2.forward(x)
        x += shortcut
        x = self.nm2.forward(x)
        shortcut = copy.deepcopy(x)
        x = self.ff.forward(x)
        x = self.dp3.forward(x)
        x += shortcut
        x = self.nm3.forward(x)
        return x

    def backward(self, x):
        x = self.nm3.backward(x)
        shortcut = copy.deepcopy(x)
        x = self.dp3.backward(x)
        x = self.ff.backward(x)
        x += shortcut
        x = self.nm2.backward(x)
        shortcut = copy.deepcopy(x)
        x = self.dp2.backward(x)
        x, k, v = self.ms2.backward(x)
        x2 = k+v
        x += shortcut
        x = self.nm1.backward(x)
        shortcut = copy.deepcopy(x)
        x = self.dp1.backward(x)
        q, k, v = self.ms1.backward(x)
        q = self.emdQ.backward(q)
        k = self.emdK.backward(k)
        v = self.emdV.backward(v)
        x1 = q+k+v+shortcut
        return x1, x2

    def zero_gradient(self):
        self.ff.zero_gradient()
        self.ms1.zero_gradient()
        self.ms2.zero_gradient()
        self.nm1.zero_gradient()
        self.nm2.zero_gradient()
        self.nm3.zero_gradient()
        self.emdQ.zero_gradient()
        self.emdK.zero_gradient()
        self.emdV.zero_gradient()

    def gradient(self):
        self.ff.gradient()
        self.ms1.gradient()
        self.ms2.gradient()
        self.nm1.gradient()
        self.nm2.gradient()
        self.nm3.gradient()
        self.emdQ.gradient()
        self.emdK.gradient()
        self.emdV.gradient()

    def adam(self, lr):
        self.ff.adam(lr)
        self.ms1.adam(lr)
        self.ms2.adam(lr)
        self.nm1.adam(lr)
        self.nm2.adam(lr)
        self.nm3.adam(lr)
        self.emdQ.adam(lr)
        self.emdK.adam(lr)
        self.emdV.adam(lr)

    def save(self, path):
        self.ff.save(path+"decff")
        self.emdQ.save(path+"decemdq")
        self.emdK.save(path+"decemdk")
        self.emdV.save(path+"decemdv")
        self.nm1.save(path+"decnm1")
        self.nm2.save(path+"decnm2")
        self.nm3.save(path+"decnm3")
        self.ms1.save(path+"decms1")
        self.ms2.save(path+"decms2")

    def load(self, path):
        self.ff.load(path+"decff")
        self.emdQ.load(path+"decemdq")
        self.emdK.load(path+"decemdk")
        self.emdV.load(path+"decemdv")
        self.nm1.load(path+"decnm1")
        self.nm2.load(path+"decnm2")
        self.nm3.load(path+"decnm3")
        self.ms1.load(path+"decms1")
        self.ms2.load(path+"decms2")
