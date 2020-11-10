from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      f = X[i].dot(W)
      f -= np.max(f)
      ll = 0.0
      tmp_dW = np.zeros(W.shape)
      for j in range(num_classes):
        ll += np.exp(f[j])
        tmp_dW[:,j] += X[i]*np.exp(f[j])
      dW[:,y[i]] -= X[i]
      loss += np.log(ll)
      loss -= f[y[i]]
      dW += tmp_dW / ll 
      
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += reg * 2.0 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    f = np.matmul(X, W)
    f -= np.max(f)
    loss = np.mean(-f[np.arange(num_train), y] + np.log(np.sum(np.exp(f), axis=1)))
    dW = np.matmul(np.exp(f.T)/np.sum(np.exp(f), axis=1), X).T
    y_matrix = np.zeros((num_train, num_classes))
    y_matrix[np.arange(num_train), y] = 1.0
    dW -= np.matmul(X.T, y_matrix)
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += reg * 2.0 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
