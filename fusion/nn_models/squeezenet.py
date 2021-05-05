from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConcatLayer
from fusion.scheduling.batch_size import get_batch_size

"""
SqueezeNet

Forrest N. Iandola, Song Han, ect., 2016
"""
batch_size = get_batch_size()

NN = Network('SqueezeNet')

NN.set_input_layer(InputLayer(3, 227, nimg=batch_size))
NN.add('conv1', ConvLayer(3, 96, 111, 7, 2, nimg=batch_size))
NN.add('pool1', PoolingLayer(96, 55, 3, strd=2, nimg=batch_size))


def add_fire(network, incp_id, sfmap, nfmaps_in, nfmaps_s1, nfmaps_e1,
             nfmaps_e3, nfmaps_c):
    ''' Add an inception module to the network. '''
    pfx = 'fire{}_'.format(incp_id)
    # squeeze1x1
    network.add(pfx + 'squeeze1x1', ConvLayer(nfmaps_in, nfmaps_s1, sfmap, 1, nimg=batch_size))

    # expand1x1
    prevs = pfx + 'squeeze1x1'
    network.add(pfx + 'expand1x1', ConvLayer(nfmaps_s1, nfmaps_e1, sfmap, 1, nimg=batch_size),
                prevs=prevs)
    # expand3x3
    network.add(pfx + 'expand3x3', ConvLayer(nfmaps_s1, nfmaps_e3, sfmap, 3, nimg=batch_size),
                prevs=prevs)

    # concat
    prevs = (pfx + 'expand1x1', pfx + 'expand3x3',)
    network.add(pfx + 'concat', ConcatLayer(nfmaps_c, sfmap, nimg=batch_size), prevs=prevs)


add_fire(NN, '2', 55, 96, 16, 64, 64, 128)
add_fire(NN, '3', 55, 128, 16, 64, 64, 128)
add_fire(NN, '4', 55, 128, 32, 128, 128, 256)
NN.add('pool4', PoolingLayer(256, 27, 3, strd=2, nimg=batch_size))
add_fire(NN, '5', 27, 256, 32, 128, 128, 256)
add_fire(NN, '6', 27, 256, 48, 192, 192, 384)
add_fire(NN, '7', 27, 384, 48, 192, 192, 384)
add_fire(NN, '8', 27, 384, 64, 256, 256, 512)
NN.add('pool8', PoolingLayer(512, 13, 3, strd=2, nimg=batch_size))
add_fire(NN, '9', 13, 512, 64, 256, 256, 512)
NN.add('conv10', ConvLayer(512, 1000, 13, 1, 1, nimg=batch_size))
NN.add('pool10', PoolingLayer(1000, 1, 13, nimg=batch_size))
