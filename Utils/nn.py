import tensorflow as tf
from math import sqrt
import logging

def countParams():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters

    return total_parameters



def count_param():
    for variable in tf.trainable_variables():
        total_parameters = 0
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
        return total_parameters


# Create some wrappers for simplicity
def conv2d_relu(x, W, b, strides=1, pre_pad=0, use_batchnorm=False, batchnorm_MA_frac=0.95):
    '''
    Conv2D wrapper.
    1) Applies zero padding in the width and height dimensions (dim 1 and 2)
    2) Applies 2d-Convolution on the padded input with bias and relu activation
    '''

    x = tf.pad(x, [[0,0],[pre_pad,pre_pad],[pre_pad,pre_pad],[0,0]], "CONSTANT") # zero padding
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    if use_batchnorm:
        x = tf.contrib.layers.batch_norm(x, batchnorm_MA_frac)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, strides=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')

def avgpool2d(x, k=2, strides=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')

def CONV2dVar(ker_w, ker_h, in_channels, out_channels, name):
    '''
    Constructor of a weight matrix for a 2d-convolutional layer
    :param ker_w: convolutional kernel width
    :param ker_h: convolutional kernel width
    :param in_channels: convolutional kernel 3rd dimension (input channels number)
    :param out_channels: number of convolutional kernels
    :param name: the name of the variable, to be assigned
    :return: the tf.Variable, , defined for an intialization from ~N(0,sqrt(2.0/num_inputs_to_neuron))
    '''
    return tf.Variable(tf.random_normal([ker_w, ker_h, in_channels, out_channels], 0, sqrt(2.0 / (in_channels * ker_w * ker_h))), name=name)

def FCVar(n_inputs, n_outputs, name):
    '''
    Constructor of a weight matrix for a fully-connected layer
    :param n_inputs:
    :param n_outputs:
    :param name:
    :return: a tf.Variable, defined for an intialization from ~N(0,sqrt(2.0/num_inputs_to_neuron))
    '''
    return tf.Variable(tf.random_normal([n_inputs, n_outputs], 0,sqrt(2.0/n_inputs)), name=name)


def BiasVar(n_biases, name):
    '''
    Constructor of a bias vector
    :param n_biases:
    :param name:
    :return: a tf.Variable, defined for an intialization to zeros
    '''
    return tf.Variable(tf.zeros([n_biases]), name=name)

def resnet_module(input,
                  weights,
                  biases,
                  module_id,
                  n_resnet_blocks,
                  n_layers_in_block,
                  filter_sizes_in_block_lst,
                  subsample_on_first_block=False,
                  use_batchnorm = True,
                  batchnorm_MA_frac = 0.95):
    '''

    Constructs a ResNet module consisting of "n_resnet_blocks" blocks. Each block
    contains "n_layers_in_block" layers with the skip-connection going from the forst to last.
    Note that following the investigation of the ResNet block order of layers - it was found that the
    original paper implied the following order of layers in a ResNet block (example for n_layers_in_block=2):

    Input --> CONV --> BN --> ReLU ----> CONV --> BN --> ADD --> ReLU
       \________________________________________________/^

    :param input: the input tensor
    :param weights: dictionary with tf.Variables. Each variable should correspond to a single convolutional layer
                    parameters set. The keyleading to this variable will be of the format "wc<module_id>.<block_id>.<layer_id>"
    :param biases: dictionary with tf.Variables. Each variable should correspond to a convolutional layer biases
                    .i.e to be a 1D array of a size equal to the number of filters of that very conv-layer. The name
                    format of the key is "bc<module_id>.<block_id>.<layer_id>"
    :param module_id: an integer used for identification of the proper weights and biases by their names.
    :param n_resnet_blocks: number of residual blocks to be cascaded
    :param n_layers_in_block : 2 for classical resnet block,
    :param filter_sizes_in_block_lst: a list with the filter numbers of every layer inside a block
                                   for example, if the bottleneck structure is desired, then
                                   the classical resnet block consists of two convolutional
                                   layers with the corresponding numbers of filters: [64,64,256]
    :param subsample_on_first_block: if True, then a subsampling of width and heigth
                                     of the input by will be performed by 2.
                                     The subsamplling is by the means of the convolutional
                                     striding in the first block out of the n_resnet_blocks

    :return:
    '''

    # construct first layer in the module (it may conatin a downsampling
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug



    ############# Layer 0 ######################################################
    # Optional sub-sampling in the first layer
    downsample_factor = 2
    strides = downsample_factor if subsample_on_first_block else 1
    pre_pad = filter_sizes_in_block_lst[0] / 2

    if subsample_on_first_block:

        # Step 1 - downsample the width and the height of the x
        x = avgpool2d(input, k=downsample_factor, strides=downsample_factor)
        #ld("x wxh was downsampled. x is now of a shape: {}".format(x.shape))

        # Step 2 - extend the 3rd dimension of the x's volume:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, x.shape[-1]]], "CONSTANT")  # zero padding
        #ld("x num-channels was extended. x is now of a shape: {}".format(x.shape))

    else:

        x = input

    weightName = "wc{}.{}.{}".format(module_id, 0, 0)
    biasName = "bc{}.{}.{}".format(module_id, 0, 0)
    output = conv2d_relu(input, weights[weightName], biases[biasName], strides=strides, pre_pad=pre_pad, use_batchnorm=use_batchnorm, batchnorm_MA_frac=batchnorm_MA_frac)


    ###### All ther rest of n_resnet_blocks*n_layers_in_block - 1) layers #######
    for block in range(n_resnet_blocks):
        for layer in range(n_layers_in_block):

            if block==0 and layer==0:
                continue # first layer was already implemented outside these loops

            pre_pad = filter_sizes_in_block_lst[layer]/2
            weightName = "wc{}.{}.{}".format(module_id, block, layer)
            biasName = "bc{}.{}.{}".format(module_id, block, layer)

            if layer == 0:
                x = output # prepare the shortcut origin

            if layer == n_layers_in_block-1: # apply the shortcut connection

                output = tf.pad(output, [[0, 0], [pre_pad, pre_pad], [pre_pad, pre_pad], [0, 0]], "CONSTANT")  # zero padding
                output = tf.nn.conv2d(output, weights[weightName], strides=[1, 1, 1, 1], padding='VALID')
                output = tf.nn.bias_add(output, biases[biasName])
                if use_batchnorm:
                    output = tf.contrib.layers.batch_norm(output, batchnorm_MA_frac)
                output = tf.add(output,x) # the holy grail of the resnet!
                output = tf.nn.relu(output)
            else:
                output = conv2d_relu(output, weights[weightName], biases[biasName], strides=1, pre_pad=pre_pad,
                                     use_batchnorm=use_batchnorm, batchnorm_MA_frac=batchnorm_MA_frac)

            #ld(" output shape is {}".format(output.shape))
    return output