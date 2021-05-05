from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-18

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet18')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 56, 3, 2, nimg=batch_size))

RES_PREV = 'pool1'

for i in range(2):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64, 64, 56, 3, nimg=batch_size))
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_br', ConvLayer(64, 64, 56, 1, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv2_br'
    NN.add('conv2_{}_res'.format(i), EltwiseLayer(64, 56, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv2_{}_b'.format(i)))
    RES_PREV = 'conv2_{}_res'.format(i)

for i in range(2):
    NN.add('conv3_{}_a'.format(i),
           ConvLayer(64, 128, 28, 3, 2, nimg=batch_size) if i == 0
           else ConvLayer(128, 128, 28, 3, nimg=batch_size))
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv3_br', ConvLayer(64, 128, 28, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv3_br'
    NN.add('conv3_{}_res'.format(i), EltwiseLayer(128, 28, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv3_{}_b'.format(i)))
    RES_PREV = 'conv3_{}_res'.format(i)

for i in range(2):
    NN.add('conv4_{}_a'.format(i),
           ConvLayer(128, 256, 14, 3, 2, nimg=batch_size) if i == 0
           else ConvLayer(256, 256, 14, 3, nimg=batch_size))
    NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 14, 3, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv4_br', ConvLayer(128, 256, 14, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv4_br'
    NN.add('conv4_{}_res'.format(i), EltwiseLayer(256, 14, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv4_{}_b'.format(i)))
    RES_PREV = 'conv4_{}_res'.format(i)

for i in range(2):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(256, 512, 7, 3, 2, nimg=batch_size) if i == 0
           else ConvLayer(512, 512, 7, 3, nimg=batch_size))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv5_br', ConvLayer(256, 512, 7, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv5_br'
    NN.add('conv5_{}_res'.format(i), EltwiseLayer(512, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv5_{}_b'.format(i)))
    RES_PREV = 'conv5_{}_res'.format(i)

NN.add('pool5', PoolingLayer(512, 1, 7, nimg=batch_size))

# NN.add('fc', FCLayer(512, 1000, nimg=batch_size))
