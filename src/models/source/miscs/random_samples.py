import numpy as np

def sample_continuous(dim, batchsize, distribution='normal', xp=np):
    if distribution == 'normal':
        return xp.random.randn(batchsize, dim).astype(xp.float32)
    elif distribution == 'uniform':
        return xp.random.uniform(batchsize, dim).astype(xp.float32)
    else:
        raise NotImplementedError
