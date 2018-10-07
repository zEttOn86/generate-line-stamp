import math
import chainer
import chainer.links as L
import chainer.functions as F


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))

class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                activation=F.relu, upsample=False):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        self.activation = activation
        self.upsample = upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.b1 = L.BatchNormalization(in_channels)
            self.b2 = L.BatchNormalization(hidden_channels)
            self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        x = upsample_conv(x, self.c_sc)
        return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
