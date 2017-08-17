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
  dim, num_class = W.shape
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_score = scores[y[i]]
    sum_score = np.sum(np.exp(scores))
    loss += (-correct_score + np.log(sum_score))
    gradient = 0.0
    for j in range(num_class):
        dW[:, j] += (np.exp(scores[j])/sum_score) * X[i]
    dW[:, y[i]] -= X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dim, num_class = W.shape
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train), y]
  loss = np.sum(-correct_scores + np.log(np.sum(np.exp(scores),axis=1)))
  temp = np.exp(scores) / np.sum(np.exp(scores),axis=1)[:,np.newaxis] # temp.shape = [num_train, num_class]
  temp[np.arange(num_train), y] -= 1
  dW = X.T.dot(temp)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

