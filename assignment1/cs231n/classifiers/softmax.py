import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # ############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    for i in xrange(num_train):
        scores = X[i].dot(W)
        escores = np.exp(scores)
        correct_class_escore = escores[y[i]]

        escores_sum = np.sum(escores)

        loss -= np.log(correct_class_escore / escores_sum)

        for j in xrange(num_classes):
            coeff = escores[j] / escores_sum
            if j == y[i]:
                coeff -= 1

            dW[:, j] += coeff * X[i].T

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    loss /= num_train

    dW /= num_train

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    # ############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores = X.dot(W)
    escores = np.exp(scores)

    escores_sums = np.sum(escores, axis=1)

    correct_percs = escores[range(num_train), y] / escores_sums
    correct_log_percs = np.log(correct_percs)

    loss -= np.sum(correct_log_percs)
    loss /= num_train

    coeffs = (escores.T / escores_sums).T
    # Subtract one off all the correct classes' values
    coeffs[range(num_train), y] -= 1

    dW = X.T.dot(coeffs)
    dW /= num_train

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

