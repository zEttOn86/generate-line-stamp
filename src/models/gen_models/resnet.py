import chainer
import chainer.links as L
import chainer.functions as F
from gen_models.resblocks import Block
from source.miscs.random_samples import sample_continuous

class ResNetGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True) #(4x4) => (8x8)
            self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True) #(8x8) => (16x16)
            self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True) #(16x16) => (32x32)
            self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True) #(32x32) => (64x64)
            self.block6 = Block(ch * 2, ch, activation=activation, upsample=True) #(64x64) => (128x128)
            self.block7 = Block(ch, ch//2, activation=activation, upsample=True) #(128x128) => (256x256)
            self.b7 = L.BatchNormalization(ch//2)
            self.l7 = L.Convolution2D(ch//2, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width)) # (Batchsize, auto, 4, 4)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))
        return h
