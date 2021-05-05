from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-152

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet152')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 56, 3, 2, nimg=batch_size))

RES_PREV = 'pool1'

for i in range(3):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64 if i == 0 else 256, 64, 56, 1, nimg=batch_size))
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3, nimg=batch_size))
    NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_br', ConvLayer(64, 256, 56, 1, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv2_br'
    NN.add('conv2_{}_res'.format(i), EltwiseLayer(256, 56, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv2_{}_c'.format(i)))
    RES_PREV = 'conv2_{}_res'.format(i)

for i in range(8):
    NN.add('conv3_{}_a'.format(i),
           ConvLayer(256, 128, 28, 1, 2, nimg=batch_size) if i == 0
           else ConvLayer(512, 128, 28, 1, nimg=batch_size))
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3, nimg=batch_size))
    NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 28, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv3_br', ConvLayer(256, 512, 28, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv3_br'
    NN.add('conv3_{}_res'.format(i), EltwiseLayer(512, 28, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv3_{}_c'.format(i)))
    RES_PREV = 'conv3_{}_res'.format(i)

for i in range(36):
    NN.add('conv4_{}_a'.format(i),
           ConvLayer(512, 256, 14, 1, 2, nimg=batch_size) if i == 0
           else ConvLayer(1024, 256, 14, 1, nimg=batch_size))
    NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 14, 3, nimg=batch_size))
    NN.add('conv4_{}_c'.format(i), ConvLayer(256, 1024, 14, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv4_br', ConvLayer(512, 1024, 14, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv4_br'
    NN.add('conv4_{}_res'.format(i), EltwiseLayer(1024, 14, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv4_{}_c'.format(i)))
    RES_PREV = 'conv4_{}_res'.format(i)

for i in range(3):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(1024, 512, 7, 1, 2, nimg=batch_size) if i == 0
           else ConvLayer(2048, 512, 7, 1, nimg=batch_size))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3, nimg=batch_size))
    NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 7, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv5_br', ConvLayer(1024, 2048, 7, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv5_br'
    NN.add('conv5_{}_res'.format(i), EltwiseLayer(2048, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv5_{}_c'.format(i)))
    RES_PREV = 'conv5_{}_res'.format(i)

NN.add('pool5', PoolingLayer(2048, 1, 7, nimg=batch_size))

NN.add('fc', FCLayer(2048, 1000, nimg=batch_size))

