#coding :utf-8
import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable
from source.miscs.random_samples import sample_continuous, sample_categorical

# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1.-dis_real))
    loss += F.mean(F.relu(1.+dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        super(Updater, self).__init__(*args, **kwargs)

    def _generate_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples

        gen = self.models['gen']
        x_fake = gen(n_gen_samples)
        return x_fake

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0])).astype('f')
        x_real = Variable(xp.asarray(x))
        return x_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        for i in range(self.n_dis):
            if i==0:
                # Update generator
                x_fake = self._generate_samples()
                dis_fake = dis(x_fake)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            # Update discriminator
            x_real = self.get_batch(xp)
            batchsize = len(x_real)
            dis_real = dis(x_real)
            x_fake = self._generate_samples(n_gen_samples=batchsize)
            dis_fake = dis(x_fake)
            x_fake.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})
