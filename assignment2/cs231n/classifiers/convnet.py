import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def affine(hidden_dim=100):
    return {
        'forward': affine_forward,
        'backward': affine_backward,
        'hidden_dim': hidden_dim
    }
def affine_relu(hidden_dim=100):
    return {
        'forward': affine_relu_forward,
        'backward': affine_relu_backward,
        'hidden_dim': hidden_dim
    }
def conv_relu(num_filters=32, filter_size=7, stride=1):
    conv_param = {'stride': stride, 'pad': (filter_size - 1) / 2}

    def _conv_relu_forward_closure(x, w, b):
        return conv_relu_forward(x, w, b, conv_param)

    return {
        'forward': _conv_relu_forward_closure,
        'backward': conv_relu_backward,
        'num_filters': num_filters,
        'filter_size': filter_size
    }
def conv_relu_pool(num_filters=32, filter_size=7, stride=1, pool_height=2, pool_width=2, pool_stride=2):
    conv_param = {'stride': stride, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}

    def _conv_relu_pool_forward_closure(x, w, b):
        return conv_relu_pool_forward(x, w, b, conv_param, pool_param)

    return {
        'forward': _conv_relu_pool_forward_closure,
        'backward': conv_relu_pool_backward,
        'num_filters': num_filters,
        'filter_size': filter_size,
        'pool_param': pool_param
    }

class ConvNet(object):
    """
    A convolutional network with an arbitrary architecture.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), conv_layers=[], hidden_layers=[],
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 use_batchnorm=False, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        # Validate inputs
        assert len(conv_layers) != 0, 'conv_layers can not be empty'
        assert len(hidden_layers) != 0, 'hidden_layers can not be empty'

        C, H, W = input_dim

        # Initialize convolutional layers
        for idx, conv_layer in enumerate(conv_layers):
            num_filters = conv_layer['num_filters']
            filter_size = conv_layer['filter_size']

            weight_label = 'W%d' % (idx + 1)
            bias_label = 'b%d' % (idx + 1)

            if idx == 0:
                channels = C
                conv_layer['output_dim'] = (num_filters, H, W)
            else:
                prev_layer = conv_layers[idx - 1]
                channels = prev_layer['num_filters']
                conv_layer['output_dim'] = (num_filters, prev_layer['output_dim'][1], prev_layer['output_dim'][2])

            self.params[weight_label] = weight_scale * np.random.randn(num_filters, channels, filter_size, filter_size)
            self.params[bias_label] = np.zeros(num_filters)

            if 'pool_param' in conv_layer:
                pool_param = conv_layer['pool_param']
                pool_height = pool_param['pool_height']
                pool_width = pool_param['pool_width']
                pool_stride = pool_param['stride']

                conv_layer['output_dim'] = (
                    conv_layer['output_dim'][0],
                    ((conv_layer['output_dim'][1] - pool_height) / pool_stride) + 1,
                    ((conv_layer['output_dim'][2] - pool_width) / pool_stride) + 1
                )

            if self.use_batchnorm:
                gamma_label = 'gamma%d' % (idx + 1)
                beta_label = 'beta%d' % (idx + 1)

                self.params[gamma_label] = np.ones(num_filters)
                self.params[beta_label] = np.zeros(num_filters)

        fc_layers = hidden_layers + [affine(num_classes)]

        # Initialize hidden fully connected layers and last fully connected layers
        for idx, fc_layer in enumerate(fc_layers):
            # If this is the first fc layer in the network, then the output is dependent
            # on that of the last conv layer
            if idx == 0:
                conv_output = conv_layers[-1]['output_dim']
                input_dim = 1
                for dim in conv_output:
                    input_dim *= dim
            else:
                input_dim = fc_layers[idx - 1]['hidden_dim']

            weight_label = 'W%d' % (idx + 1 + len(conv_layers))
            bias_label = 'b%d' % (idx + 1 + len(conv_layers))

            output_dim = fc_layer['hidden_dim']

            self.params[weight_label] = weight_scale * np.random.randn(input_dim, output_dim)
            self.params[bias_label] = np.zeros(output_dim)

            if self.use_batchnorm and (idx + 1 + len(conv_layers)) < len(conv_layers) + len(hidden_layers) + 1:
                gamma_label = 'gamma%d' % (idx + 1 + len(conv_layers))
                beta_label = 'beta%d' % (idx + 1 + len(conv_layers))

                self.params[gamma_label] = np.ones(output_dim)
                self.params[beta_label] = np.zeros(output_dim)

        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.layers = [None] + self.conv_layers + self.fc_layers

        self.num_layers = len(conv_layers) + len(fc_layers)
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in xrange(self.num_layers)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)

        if self.use_batchnorm:
            mode = 'test' if y is None else 'train'
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # Each layer's cache consists of its output, forward cache, and batchnorm cache
        layer_cache = [(X, None, None)]
        for layer in xrange(1, self.num_layers + 1):
            conv = self.layers[layer]

            weight_label = 'W%d' % layer
            bias_label = 'b%d' % layer

            x = layer_cache[layer - 1][0]
            w = self.params[weight_label]
            b = self.params[bias_label]

            bn_cache = None
            z, z_cache = conv['forward'](x, w, b)
            if self.use_batchnorm and layer != self.num_layers:
                gamma_label = 'gamma%d' % layer
                beta_label = 'beta%d' % layer

                bn_param, gamma, beta = self.bn_params[layer - 1], self.params[gamma_label], self.params[beta_label]
                if layer <= len(self.conv_layers):
                    z, bn_cache = spatial_batchnorm_forward(z, gamma, beta, bn_param)
                else:
                    z, bn_cache = batchnorm_forward(z, gamma, beta, bn_param)

            layer_cache.append((z, z_cache, bn_cache))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return layer_cache[-1][0]

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dL = softmax_loss(layer_cache[-1][0], y)
        if self.reg > 0:
            loss += 0.5 * self.reg * np.sum(np.sum(v * v) for k, v in self.params.iteritems() if 'W' in k)

        dout = dL
        for layer in xrange(self.num_layers, 0, -1):
            layer_obj = self.layers[layer]

            weight_label = 'W%d' % layer
            bias_label = 'b%d' % layer

            # Retrieve cached values for the layer
            cache = layer_cache[layer]

            # If it is not the output layer, we need to take into account our ReLU
            # to decide where gradient backpropagated to.
            if self.use_batchnorm and layer != self.num_layers:
                gamma_label = 'gamma%d' % layer
                beta_label = 'beta%d' % layer
                if layer <= len(self.conv_layers):
                    dout, grads[gamma_label], grads[beta_label] = spatial_batchnorm_backward(dout, cache[2])
                else:
                    dout, grads[gamma_label], grads[beta_label] = batchnorm_backward_alt(dout, cache[2])

            # Calculate gradients for this layer
            dz, grads[weight_label], grads[bias_label] = layer_obj['backward'](dout, cache[1])

            if self.reg > 0:
                weight = self.params[weight_label]
                grads[weight_label] += self.reg * weight

            dout = dz

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
