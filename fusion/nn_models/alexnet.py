from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConcatLayer
from fusion.scheduling.batch_size import get_batch_size

batch_size = get_batch_size()

'''
AlexNet

Krizhevsky, Sutskever, and Hinton, 2012
'''

# NN = Network('AlexNet')
#
# NN.set_input_layer(InputLayer(3, 227, nimg=batch_size))
#
# NN.add('conv1_a', ConvLayer(3, 48, 55, 11, 4, nimg=batch_size), prevs=(NN.INPUT_LAYER_KEY,))
# NN.add('conv1_b', ConvLayer(3, 48, 55, 11, 4, nimg=batch_size), prevs=(NN.INPUT_LAYER_KEY,))
# NN.add('pool1_a', PoolingLayer(48, 27, 3, strd=2, nimg=batch_size), prevs=('conv1_a',))
# NN.add('pool1_b', PoolingLayer(48, 27, 3, strd=2, nimg=batch_size), prevs=('conv1_b',))
# # Norm layer is ignored.
#
# NN.add('conv2_a', ConvLayer(48, 128, 27, 5, nimg=batch_size), prevs=('pool1_a',))
# NN.add('conv2_b', ConvLayer(48, 128, 27, 5, nimg=batch_size), prevs=('pool1_b',))
# NN.add('pool2_a', PoolingLayer(128, 13, 3, strd=2, nimg=batch_size), prevs=('conv2_a',))
# NN.add('pool2_b', PoolingLayer(128, 13, 3, strd=2, nimg=batch_size), prevs=('conv2_b',))
# # Norm layer is ignored.
#
# NN.add('conv2_concat', ConcatLayer(256, 13, nimg=batch_size), prevs=('pool2_a', 'pool2_b'))
#
# NN.add('conv3_a', ConvLayer(256, 192, 13, 3, nimg=batch_size), prevs=('conv2_concat',))
# NN.add('conv3_b', ConvLayer(256, 192, 13, 3, nimg=batch_size), prevs=('conv2_concat',))
# NN.add('conv4_a', ConvLayer(192, 192, 13, 3, nimg=batch_size), prevs=('conv3_a',))
# NN.add('conv4_b', ConvLayer(192, 192, 13, 3, nimg=batch_size), prevs=('conv3_b',))
# NN.add('conv5_a', ConvLayer(192, 128, 13, 3, nimg=batch_size), prevs=('conv4_a',))
# NN.add('conv5_b', ConvLayer(192, 128, 13, 3, nimg=batch_size), prevs=('conv4_b',))
# NN.add('pool3_a', PoolingLayer(128, 6, 3, strd=2, nimg=batch_size), prevs=('conv5_a',))
# NN.add('pool3_b', PoolingLayer(128, 6, 3, strd=2, nimg=batch_size), prevs=('conv5_b',))

# NN.add('conv5_concat', ConcatLayer(256, 6, nimg=batch_size), prevs=('pool3_a', 'pool3_b'))

# NN.add('fc1', FCLayer(256, 4096, 6, nimg=batch_size), prevs=('conv5_concat',))
# NN.add('fc2', FCLayer(4096, 4096, nimg=batch_size))
# NN.add('fc3', FCLayer(4096, 1000, nimg=batch_size))


NN = Network('AlexNet')

NN.set_input_layer(InputLayer(3, 227, nimg=batch_size))

NN.add('conv1_a', ConvLayer(3, 48, 55, 11, 4, nimg=batch_size), prevs=(NN.INPUT_LAYER_KEY,))
NN.add('conv1_b', ConvLayer(3, 48, 55, 11, 4, nimg=batch_size), prevs=(NN.INPUT_LAYER_KEY,))
NN.add('pool1_a', PoolingLayer(48, 27, 3, strd=2, nimg=batch_size), prevs=('conv1_a',))
NN.add('pool1_b', PoolingLayer(48, 27, 3, strd=2, nimg=batch_size), prevs=('conv1_b',))
# Norm layer is ignored.

NN.add('conv2_a', ConvLayer(48, 128, 27, 5, nimg=batch_size), prevs=('pool1_a',))
NN.add('conv2_b', ConvLayer(48, 128, 27, 5, nimg=batch_size), prevs=('pool1_b',))
NN.add('pool2_a', PoolingLayer(128, 13, 3, strd=2, nimg=batch_size), prevs=('conv2_a',))
NN.add('pool2_b', PoolingLayer(128, 13, 3, strd=2, nimg=batch_size), prevs=('conv2_b',))
# Norm layer is ignored.

NN.add('conv2_concat', ConcatLayer(256, 13, nimg=batch_size), prevs=('pool2_a', 'pool2_b'))

NN.add('conv3_a', ConvLayer(256, 192, 13, 3, nimg=batch_size), prevs=('conv2_concat',))
NN.add('conv3_b', ConvLayer(256, 192, 13, 3, nimg=batch_size), prevs=('conv2_concat',))
NN.add('conv4_a', ConvLayer(192, 192, 13, 3, nimg=batch_size), prevs=('conv3_a',))
NN.add('conv4_b', ConvLayer(192, 192, 13, 3, nimg=batch_size), prevs=('conv3_b',))
NN.add('conv5_a', ConvLayer(192, 128, 13, 3, nimg=batch_size), prevs=('conv4_a',))
NN.add('conv5_b', ConvLayer(192, 128, 13, 3, nimg=batch_size), prevs=('conv4_b',))
NN.add('pool3_a', PoolingLayer(128, 6, 3, strd=2, nimg=batch_size), prevs=('conv5_a',))
NN.add('pool3_b', PoolingLayer(128, 6, 3, strd=2, nimg=batch_size), prevs=('conv5_b',))