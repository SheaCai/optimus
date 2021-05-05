from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer
from fusion.scheduling.batch_size import get_batch_size

"""
MobileNet

Andrew G. Howard, Menglong Zhu, Bo Chen, ect., 2017
"""
batch_size = get_batch_size()

NN = Network('MobileNet')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))
NN.add('conv1', ConvLayer(3, 32, 112, 3, 2, nimg=batch_size))

NN.add('conv2_1_dw', DWConvLayer(32, 112, 3, 1, nimg=batch_size))
NN.add('conv2_1_pw', ConvLayer(32, 64, 112, 1, 1, nimg=batch_size))

NN.add('conv2_2_dw', DWConvLayer(64, 56, 3, 2, nimg=batch_size))
NN.add('conv2_2_pw', ConvLayer(64, 128, 56, 1, 1, nimg=batch_size))

NN.add('conv3_1_dw', DWConvLayer(128, 56, 3, 1, nimg=batch_size))
NN.add('conv3_1_pw', ConvLayer(128, 128, 56, 1, 1, nimg=batch_size))

NN.add('conv3_2_dw', DWConvLayer(128, 28, 3, 2, nimg=batch_size))
NN.add('conv3_2_pw', ConvLayer(128, 256, 28, 1, 1, nimg=batch_size))

NN.add('conv4_1_dw', DWConvLayer(256, 28, 3, 1, nimg=batch_size))
NN.add('conv4_1_pw', ConvLayer(256, 256, 28, 1, 1, nimg=batch_size))

NN.add('conv4_2_dw', DWConvLayer(256, 14, 3, 2, nimg=batch_size))
NN.add('conv4_2_pw', ConvLayer(256, 512, 14, 1, 1, nimg=batch_size))

for i in range(1, 6):
    NN.add('conv5_{}_dw'.format(i), DWConvLayer(512, 14, 3, 1, nimg=batch_size))
    NN.add('conv5_{}_pw'.format(i), ConvLayer(512, 512, 14, 1, 1, nimg=batch_size))

NN.add('conv5_6_dw', DWConvLayer(512, 7, 3, 2, nimg=batch_size))
NN.add('conv5_6_pw', ConvLayer(512, 1024, 7, 1, 1, nimg=batch_size))

NN.add('conv6_dw', DWConvLayer(1024, 7, 3, 1, nimg=batch_size))
NN.add('conv6_pw', ConvLayer(1024, 1024, 7, 1, 1, nimg=batch_size))

NN.add('pool6', PoolingLayer(1024, 1, 7, nimg=batch_size))
NN.add('fc', FCLayer(1024, 1000, nimg=batch_size))

