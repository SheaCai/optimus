from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConcatLayer
from fusion.scheduling.batch_size import get_batch_size

"""
GoogLeNet

ILSVRC 2014
"""
batch_size = get_batch_size()

NN = Network('GoogLeNet')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 56, 3, strd=2, nimg=batch_size))
# Norm layer is ignored.

NN.add('conv2_3x3_reduce', ConvLayer(64, 64, 56, 1, nimg=batch_size))
NN.add('conv2_3x3', ConvLayer(64, 192, 56, 3, nimg=batch_size))
# Norm layer is ignored.
NN.add('pool2', PoolingLayer(192, 28, 3, strd=2, nimg=batch_size))


def add_inception(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1, nimg=batch_size),
                prevs=prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1, nimg=batch_size),
                prevs=prevs)
    network.add(pfx + '3x3', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3, nimg=batch_size))
    # 5x5 branch.
    network.add(pfx + '5x5_reduce', ConvLayer(nfmaps_in, nfmaps_5r, sfmap, 1, nimg=batch_size),
                prevs=prevs)
    network.add(pfx + '5x5', ConvLayer(nfmaps_5r, nfmaps_5, sfmap, 5, nimg=batch_size))
    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1, nimg=batch_size),
                prevs=prevs)
    # Merge branches.
    return pfx + '1x1', pfx + '3x3', pfx + '5x5', pfx + 'pool_proj'


_PREVS = ('pool2',)

# Inception 3.
_PREVS = add_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32,
                       prevs=_PREVS)
NN.add('inception_3a', ConcatLayer(256, 28, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_3a'
_PREVS = add_inception(NN, '3b', 28, 256, 128, 128, 192, 32, 96, 64,
                       prevs=_PREVS)
NN.add('inception_3b', ConcatLayer(480, 28, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_3b'
NN.add('pool3', PoolingLayer(480, 14, 3, strd=2, nimg=batch_size), prevs=_PREVS)
_PREVS = ('pool3',)

# Inception 4.
_PREVS = add_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64,
                       prevs=_PREVS)
NN.add('inception_4a', ConcatLayer(512, 14, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_4a'
_PREVS = add_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64,
                       prevs=_PREVS)
NN.add('inception_4b', ConcatLayer(512, 14, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_4b'
_PREVS = add_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64,
                       prevs=_PREVS)
NN.add('inception_4c', ConcatLayer(512, 14, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_4c'
_PREVS = add_inception(NN, '4d', 14, 512, 112, 144, 288, 32, 64, 64,
                       prevs=_PREVS)
NN.add('inception_4d', ConcatLayer(528, 14, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_4d'
_PREVS = add_inception(NN, '4e', 14, 528, 256, 160, 320, 32, 128, 128,
                       prevs=_PREVS)
NN.add('inception_4e', ConcatLayer(832, 14, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_4e'

NN.add('pool4', PoolingLayer(832, 7, 3, strd=2, nimg=batch_size), prevs=_PREVS)
_PREVS = ('pool4',)

# Inception 5.
_PREVS = add_inception(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128,
                       prevs=_PREVS)
NN.add('inception_5a', ConcatLayer(832, 7, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_5a'
_PREVS = add_inception(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128,
                       prevs=_PREVS)
NN.add('inception_5b', ConcatLayer(1024, 7, nimg=batch_size), prevs=_PREVS)
_PREVS = 'inception_5b'

NN.add('pool5', PoolingLayer(1024, 1, 7, nimg=batch_size), prevs=_PREVS)

NN.add('fc', FCLayer(1024, 1000, nimg=batch_size))

