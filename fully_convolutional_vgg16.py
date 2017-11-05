# This model is a modified version of VGG16 [1] to work in the fully
# convolutional manner.
#
# [1]: https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
#
# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import pickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

# BGR Mean
mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)

def subtract_mean(images, mean):
    images -= mean.reshape((1, 3, 1, 1))

def load_weights(weight_file, last_layer, use_fully_convolutional=False): 
    model = pickle.load(open(weight_file, 'rb'), encoding='bytes')

    params = model[b'param values']
    if use_fully_convolutional:
        params[-6] = params[-6].reshape((512, 7, 7, -1)).transpose(3, 0, 1, 2)
        params[-4] = params[-4].reshape((4096, 1, 1, -1)).transpose(3, 0, 1, 2)
        params[-2] = params[-2].reshape((4096, 1, 1, -1)).transpose(3, 0, 1, 2)
    lasagne.layers.set_all_param_values(last_layer, params)


def build_model(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(
        net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def build_fully_convolutional_model(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, None, None), input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = ConvLayer(
        net['pool5'], 4096, 7, pad=0, flip_filters=False)

    net['fc7'] = ConvLayer(
        net['fc6'], 4096, 1, pad=0, flip_filters=False)

    net['fc8'] = ConvLayer(
        net['fc7'], 1000, 1, pad=0, flip_filters=False)
    net['fc8'] = FlattenLayer(net['fc8'])
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


if __name__ == '__main__':
    input_var = T.ftensor4()
    net1 = build_model(input_var)
    load_weights('../vgg16.pkl', net1['prob'])
    net2 = build_fully_convolutional_model(input_var)
    load_weights('../vgg16.pkl', net2['prob'], use_fully_convolutional=True)

    vgg16_fn = theano.function([input_var], lasagne.layers.get_output(net1['prob']))
    vgg16_fully_conv_fn = theano.function([input_var], lasagne.layers.get_output(net2['prob']))

    fake_img = np.random.random((1, 3, 224, 224)).astype(np.float32)
    subtract_mean(fake_img, mean)
    print(np.argmax(vgg16_fn(fake_img)), np.argmax(vgg16_fully_conv_fn(fake_img)))

