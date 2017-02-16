import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    # ###########################################################################
    # Initialize the weights and biases of the two-layer net. Weights          #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    # ###########################################################################
    # Implement the forward pass for the two-layer net, computing the          #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    a1, a1_cache = affine_forward(X, W1, b1)
    z1, z1_cache = relu_forward(a1)

    z2, z2_cache = affine_forward(z1, W2, b2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return z2

    grads = {}
    ############################################################################
    # Implement the backward pass for the two-layer net. Store the loss        #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, dz2 = softmax_loss(z2, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    dz1, grads['W2'], grads['b2'] = affine_backward(dz2, z2_cache)
    grads['W2'] += self.reg * W2

    dz1 = relu_backward(dz1, z1_cache)
    _, grads['W1'], grads['b1'] = affine_backward(dz1, a1_cache)
    grads['W1'] += self.reg * W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # Initialize the parameters of the network, storing all values in          #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    dims = [input_dim] + hidden_dims + [num_classes]
    io_dims = zip(dims, dims[1:])

    for idx, layer in enumerate(io_dims):
      input_dim, output_dim = layer

      weight = weight_scale * np.random.randn(input_dim, output_dim)
      bias = np.zeros(output_dim)

      weight_label = 'W%d' % (idx + 1)
      bias_label = 'b%d' % (idx + 1)

      self.params[weight_label] = weight
      self.params[bias_label] = bias

      if self.use_batchnorm and idx + 1 != len(io_dims):
        gamma_label = 'gamma%d' % (idx + 1)
        beta_label = 'beta%d' % (idx + 1)

        self.params[gamma_label] = np.ones(output_dim)
        self.params[beta_label] = np.zeros(output_dim)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[1] to the forward pass
    # of the first batch normalization layer, self.bn_params[2] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    ############################################################################
    # Implement the forward pass for the fully-connected net, computing        #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    # Store each layer's cache values in this list. Layer 0 will be a
    # sentinel and it's values are the identity on the input.
    #
    # key: (int) layer number
    # value: (layer_output, a_cache, z_cache)
    layer_cache = [(X, X, X, None)]
    for layer in xrange(1, self.num_layers + 1):
      weight_label = 'W%d' % layer
      bias_label = 'b%d' % layer

      weight = self.params[weight_label]
      bias = self.params[bias_label]

      # Get the previous layer's output which is this layer's input
      input_z, _, _, _ = layer_cache[layer-1]

      a, a_cache = affine_forward(input_z, weight, bias)

      # If we are in a hidden layer, we need to pass it through our ReLU
      bn_cache = None
      if layer != self.num_layers:
        if self.use_batchnorm:
          gamma_label = 'gamma%d' % layer
          beta_label = 'beta%d' % layer

          bn_param, gamma, beta = self.bn_params[layer-1], self.params[gamma_label], self.params[beta_label]
          a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)

        z, z_cache = relu_forward(a)
      # Otherwise we just pass through with the identity
      else:
        z, z_cache = a, a

      layer_cache.append((z, a_cache, z_cache, bn_cache))


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return layer_cache[-1][0]

    loss, grads = 0.0, {}
    ############################################################################
    # Implement the backward pass for the fully-connected net. Store the       #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, dL = softmax_loss(layer_cache[-1][0], y)
    # Take into account regularization
    if self.reg > 0:
      loss += 0.5 * self.reg *  np.sum(np.sum(v * v) for k, v in self.params.iteritems() if 'W' in k)

    dout = dL
    for layer in xrange(self.num_layers, 0, -1):
      weight_label = 'W%d' % layer
      bias_label = 'b%d' % layer

      # Retrieve cached values for the layer
      _, a_cache, z_cache, bn_cache = layer_cache[layer]

      # If it is not the output layer, we need to take into account our ReLU
      # to decide where gradient backpropagated to.
      if layer != self.num_layers:
        dout = relu_backward(dout, z_cache)
        if self.use_batchnorm:
          gamma_label = 'gamma%d' % layer
          beta_label = 'beta%d' % layer
          dout, grads[gamma_label], grads[beta_label] = batchnorm_backward_alt(dout, bn_cache)

      # Calculate gradients for this layer
      dz, grads[weight_label], grads[bias_label] = affine_backward(dout, a_cache)

      # Take into account regularization
      if self.reg > 0:
        weight = self.params[weight_label]
        grads[weight_label] += self.reg * weight

      dout = dz

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
