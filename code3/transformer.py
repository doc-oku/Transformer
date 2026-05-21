from encoder import Encoder
from decoder import Decoder
from embedding import Embedding
from position import PositionEncoding
from softmax import SoftMax3d


class Transformer:
    def __init__(self, jp_vocab_size):
        d_model = 384
        self.emd1 = Embedding(d_model)
        self.pos1 = PositionEncoding()
        self.enc1 = Encoder(d_model)
        self.enc2 = Encoder(d_model)
        self.enc3 = Encoder(d_model)
        self.enc4 = Encoder(d_model)
        self.enc5 = Encoder(d_model)
        self.enc6 = Encoder(d_model)

        self.emd2 = Embedding(d_model)
        self.pos2 = PositionEncoding()
        self.dec1 = Decoder(d_model)
        self.dec2 = Decoder(d_model)
        self.dec3 = Decoder(d_model)
        self.dec4 = Decoder(d_model)
        self.dec5 = Decoder(d_model)
        self.dec6 = Decoder(d_model)
        self.emd3 = Embedding(jp_vocab_size)
        self.sm = SoftMax3d()  
        #self.load()

    def forward1(self, x):
        x = self.emd1.forward(x)
        x = self.pos1.forward(x)
        x = self.enc1.forward(x)
        x = self.enc2.forward(x)
        x = self.enc3.forward(x)
        x = self.enc4.forward(x)
        x = self.enc5.forward(x)
        self.out1 = self.enc6.forward(x)
        
    def forward2(self, x):
        x = self.emd2.forward(x)
        x = self.pos2.forward(x)
        x = self.dec1.forward(x, self.out1)
        x = self.dec2.forward(x, self.out1)
        x = self.dec3.forward(x, self.out1)
        x = self.dec4.forward(x, self.out1)
        x = self.dec5.forward(x, self.out1)
        x = self.dec6.forward(x, self.out1)
        x = self.emd3.forward(x)
        x = self.sm.forward(x)
        return x

    def backward(self, x):   
        x = self.sm.backward(x)
        x = self.emd3.backward(x)
        x, x6 = self.dec6.backward(x)
        x, x5 = self.dec5.backward(x)
        x, x4 = self.dec4.backward(x)
        x, x3 = self.dec3.backward(x)
        x, x2 = self.dec2.backward(x)
        x, x1 = self.dec1.backward(x)
        x = self.pos2.backward(x)
        self.emd2.backward(x)

        x = x1+x2+x3+x4+x5+x6
        x = self.enc6.backward(x)
        x = self.enc5.backward(x)
        x = self.enc4.backward(x)
        x = self.enc3.backward(x)
        x = self.enc2.backward(x)
        x = self.enc1.backward(x)
        x = self.pos1.backward(x)
        self.emd1.backward(x)

    def zero_gradient(self):
        self.enc1.zero_gradient()
        self.enc2.zero_gradient()
        self.enc3.zero_gradient()
        self.enc4.zero_gradient()
        self.enc5.zero_gradient()
        self.enc6.zero_gradient()
        self.emd1.zero_gradient()

        self.dec1.zero_gradient()
        self.dec2.zero_gradient()
        self.dec3.zero_gradient()
        self.dec4.zero_gradient()
        self.dec5.zero_gradient()
        self.dec6.zero_gradient()
        self.emd2.zero_gradient()
        self.emd3.zero_gradient()

    def gradient(self):
        self.enc1.gradient()
        self.enc2.gradient()
        self.enc3.gradient()
        self.enc4.gradient()
        self.enc5.gradient()
        self.enc6.gradient()
        self.emd1.gradient()

        self.dec1.gradient()
        self.dec2.gradient()
        self.dec3.gradient()
        self.dec4.gradient()
        self.dec5.gradient()
        self.dec6.gradient()
        self.emd2.gradient()
        self.emd3.gradient()

    def adam(self, lr):
        self.enc1.adam(lr)
        self.enc2.adam(lr)
        self.enc3.adam(lr)
        self.enc4.adam(lr)
        self.enc5.adam(lr)
        self.enc6.adam(lr)
        self.emd1.adam(lr)

        self.dec1.adam(lr)
        self.dec2.adam(lr)
        self.dec3.adam(lr)
        self.dec4.adam(lr)
        self.dec5.adam(lr)
        self.dec6.adam(lr)
        self.emd2.adam(lr)
        self.emd3.adam(lr)

    def save(self):
        self.enc1.save("enc1")
        self.enc2.save("enc2")
        self.enc3.save("enc3")
        self.enc4.save("enc4")
        self.enc5.save("enc5")
        self.enc6.save("enc6")
        self.emd1.save("emd1")

        self.dec1.save("dec1")
        self.dec2.save("dec2")
        self.dec3.save("dec3")
        self.dec4.save("dec4")
        self.dec5.save("dec5")
        self.dec6.save("dec6")
        self.emd2.save("emd2")
        self.emd3.save("emd3")

    def load(self):
        self.enc1.load("enc1")
        self.enc2.load("enc2")
        self.enc3.load("enc3")
        self.enc4.load("enc4")
        self.enc5.load("enc5")
        self.enc6.load("enc6")
        self.emd1.load("emd1")

        self.dec1.load("dec1")
        self.dec2.load("dec2")
        self.dec3.load("dec3")
        self.dec4.load("dec4")
        self.dec5.load("dec5")
        self.dec6.load("dec6")
        self.emd2.load("emd2")
        self.emd3.load("emd3")

