import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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

    # weight = weight_scale * np.random.randn(input_dim, output_dim)
    # bias = np.zeros(output_dim)

    C, H, W = input_dim

    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)

    # We pad the input so the output of our convolution preserves the height and width
    # followed by a 2x2 pool which reduces everything by 75%
    flattened_conv_output = num_filters * H * W / 4
    self.params['W2'] = weight_scale * np.random.randn(flattened_conv_output, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    if self.use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)

      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)

      self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    if self.use_batchnorm:
      mode = 'test' if y is None else 'train'
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    ############################################################################
    # Implement the forward pass for the three-layer convolutional net,        #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    z1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    if self.use_batchnorm:
      bn_param, gamma, beta = self.bn_params[0], self.params['gamma1'], self.params['beta1']
      z1, bn_cache1 = spatial_batchnorm_forward(z1, gamma, beta, bn_param)

    z2, cache2 = affine_relu_forward(z1, W2, b2)

    if self.use_batchnorm:
      bn_param, gamma, beta = self.bn_params[1], self.params['gamma2'], self.params['beta2']
      z2, bn_cache2 = batchnorm_forward(z2, gamma, beta, bn_param)

    z3, cache3 = affine_forward(z2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return z3
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the three-layer convolutional net,       #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dL = softmax_loss(z3, y)

    dout = dL

    dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)

    if self.use_batchnorm:
      dout, grads['gamma2'], grads['beta2'] = batchnorm_backward_alt(dout, bn_cache2)

    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache2)

    if self.use_batchnorm:
      dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, bn_cache1)

    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache1)

    if self.reg > 0:
      loss += 0.5 * self.reg * np.sum(np.sum(w * w) for w in [W1, W2, W3])

      grads['W3'] += self.reg * W3
      grads['W2'] += self.reg * W2
      grads['W1'] += self.reg * W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
