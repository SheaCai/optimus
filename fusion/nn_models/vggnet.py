from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer
from fusion.scheduling.batch_size import get_batch_size

"""
VGGNet-16

Simonyan and Zisserman, 2014
"""
batch_size = get_batch_size()
NN = Network('VGG')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 224, 3, nimg=batch_size))
NN.add('conv2', ConvLayer(64, 64, 224, 3, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 112, 2, nimg=batch_size))

NN.add('conv3', ConvLayer(64, 128, 112, 3, nimg=batch_size))
NN.add('conv4', ConvLayer(128, 128, 112, 3, nimg=batch_size))
NN.add('pool2', PoolingLayer(128, 56, 2, nimg=batch_size))

NN.add('conv5', ConvLayer(128, 256, 56, 3, nimg=batch_size))
NN.add('conv6', ConvLayer(256, 256, 56, 3, nimg=batch_size))
NN.add('conv7', ConvLayer(256, 256, 56, 3, nimg=batch_size))
NN.add('pool3', PoolingLayer(256, 28, 2, nimg=batch_size))

NN.add('conv8', ConvLayer(256, 512, 28, 3, nimg=batch_size))
NN.add('conv9', ConvLayer(512, 512, 28, 3, nimg=batch_size))
NN.add('conv10', ConvLayer(512, 512, 28, 3, nimg=batch_size))
NN.add('pool4', PoolingLayer(512, 14, 2, nimg=batch_size))

NN.add('conv11', ConvLayer(512, 512, 14, 3, nimg=batch_size))
NN.add('conv12', ConvLayer(512, 512, 14, 3, nimg=batch_size))
NN.add('conv13', ConvLayer(512, 512, 14, 3, nimg=batch_size))
NN.add('pool5', PoolingLayer(512, 7, 2, nimg=batch_size))

# NN.add('fc1', FCLayer(512, 4096, 7, nimg=batch_size))
# NN.add('fc2', FCLayer(4096, 4096, nimg=batch_size))
# NN.add('fc3', FCLayer(4096, 1000, nimg=batch_size))

