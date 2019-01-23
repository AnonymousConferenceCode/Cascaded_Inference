import numpy as np
import tensorflow as tf

"""
Maxout OP from https://arxiv.org/abs/1302.4389
Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.
Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def max_out_inception(inputs, num_units_list, axis=None):
    outputs = []    
    for num_units in num_units_list:
        outputs += [max_out(inputs, num_units, axis)]
    outputs = tf.concat(1, outputs)
    return outputs       

# Create some wrappers for simplicity
def conv2d_maxout(x, W, b, maxout_units, strides=1, pre_pad=0, max_kernel_norm=None, use_batchnorm=False, batchnorm_MA_frac=0.95):
    '''
    Conv2D wrapper, with bias and maxout activation
    It does the following operations:
        1) applies a padding using pre_pad (scalar) padding in width and height
           This effectively increases the images width with additional 2*pre_pad
            pixels (the same for the height)
        2) applie 2d-Convolution using the filters W and biases b
        3) applies maxout operator on the output of the convolution
            this outputs the volume with the same width and height as 
            the convolution, BUT the number of channels will be reduced
            to "maxout_units" value.
    '''
    if max_kernel_norm != None:
        W = tf.clip_by_norm(W, max_kernel_norm, axes=[0,1,2])
    x = tf.pad(x, [[0,0],[pre_pad,pre_pad],[pre_pad,pre_pad],[0,0]], "CONSTANT") # zero padding
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    if use_batchnorm:
        x = tf.contrib.layers.batch_norm(x, batchnorm_MA_frac)
    return max_out(x, maxout_units)



