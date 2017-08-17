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
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
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
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)
    self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
    self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    pass
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
    scores = None
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    reg = self.reg
    first_level_scores, first_level_cache = affine_relu_forward(X, W1, b1)
    second_level_scores, second_level_cache = affine_forward(first_level_scores, W2, b2)
    scores = np.copy(second_level_scores)
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    dx2, dW2, db2 = affine_backward(dscores, second_level_cache)
    grads['W2'] = dW2 + reg * W2
    grads['b2'] = db2
    dx1, dW1, db1 = affine_relu_backward(dx2, first_level_cache)
    grads['W1'] = dW1 + reg * W1
    grads['b1'] = db1
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
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

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
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
    num_layers = self.num_layers
    
    self.params['b%d' % num_layers] = np.zeros(num_classes)
    self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
    self.params['W%d' % num_layers] = np.random.normal(0, weight_scale, (hidden_dims[num_layers - 2], num_classes))
    if use_batchnorm:
        self.params['gamma1'] = np.ones(input_dim)
        self.params['beta1'] = np.zeros(input_dim)
    for i in xrange(1, num_layers):
        bs = 'b%d' % i
        self.params[bs] = np.zeros(hidden_dims[i - 1])
        if use_batchnorm:
            gammas = 'gamma%d' % i
            betas = 'beta%d' % i
            self.params[gammas] = np.ones(hidden_dims[i - 1])
            self.params[betas] = np.zeros(hidden_dims[i - 1])
        if i == 1:
            continue
        ws = 'W%d' % i
        self.params[ws] = np.random.normal(0, weight_scale, (hidden_dims[i - 2], hidden_dims[i - 1]))
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
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
    pass
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
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
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

    scores = np.copy(X)
    num_layers = self.num_layers
    cache = {} 
    for i in xrange(1, num_layers):
        W = self.params['W%d' % i]
        b = self.params['b%d' % i]
        if self.use_batchnorm and self.use_dropout:
            # case when use BOTH dropout AND batchnorm 
            print 'hehe'
        if self.use_dropout and not self.use_batchnorm:
            # case when use dropout and NOT batchnorm 
            scores, cache['cache_layer_%d' % i] = affine_relu_dropout_forward(scores, W, b, self.dropout_param)
        if self.use_batchnorm and not self.use_dropout:
            # case when use batchnorm and NOT dropout
            gamma = self.params['gamma%d' % i]
            beta = self.params['beta%d' % i]
            scores, cache['cache_layer_%d' % i] = affine_batchnorm_relu_forward(scores, W, b, gamma, beta, self.bn_params[i - 1])
        if not self.use_batchnorm and not self.use_dropout: 
            # case when use nothing, simply affine and relu 
            scores, cache['cache_layer_%d' % i] = affine_relu_forward(scores, W, b)
            

    W_last_layer = self.params['W%d' % num_layers]
    b_last_layer = self.params['b%d' % num_layers]
    # whatever method used before, including batchnorm and dropout or not, the last layer implementation is always the same 
    scores, cache['cache_layer_%d' % num_layers] = affine_forward(scores, W_last_layer, b_last_layer)
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
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
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    reg = self.reg 
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * reg * np.sum(W_last_layer * W_last_layer)
    # whatever forwarding method used, the first back propagation is always the same 
    dx, dW_last_layer, db_last_layer = affine_backward(dscores, cache['cache_layer_%d' % num_layers])
    grads['W%d' % num_layers] = dW_last_layer + reg * W_last_layer
    grads['b%d' % num_layers] = db_last_layer
    
    for i in reversed(xrange(1, num_layers)):
        W = self.params['W%d' % i]
        loss += 0.5 * reg * np.sum(W * W)
        if self.use_batchnorm and self.use_dropout:
            print 'hehe'
        if self.use_dropout and not self.use_batchnorm:
            dx, dW, db = affine_relu_dropout_backward(dx, cache['cache_layer_%d' % i])
        if self.use_batchnorm and not self.use_dropout:
            dx, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dx, cache['cache_layer_%d' % i])
            grads['gamma%d' % i] = dgamma
            grads['beta%d' % i] = dbeta
        if not self.use_batchnorm and not self.use_dropout :
            dx, dW, db = affine_relu_backward(dx, cache['cache_layer_%d' % i])
            
        grads['W%d' % i] = dW + reg * W
        grads['b%d' % i] = db
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
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
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    affine_score, affine_cache = affine_forward(x, w, b)
    batchnorm_score, batchnorm_cache = batchnorm_forward(affine_score, gamma, beta, bn_param)
    relu_score, relu_cache = relu_forward(batchnorm_score)
    final_score = np.copy(relu_score)
    cache = (affine_cache, batchnorm_cache, relu_cache)
    return final_score, cache

def affine_batchnorm_relu_backward(dout, cache):
    affine_cache, batchnorm_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dbatchnorm, dgamma, dbeta = batchnorm_backward(drelu, batchnorm_cache)
    daffine, dw, db = affine_backward(dbatchnorm, affine_cache)
    dx = np.copy(daffine)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_param):
    affine_relu_score, affine_relu_cache = affine_relu_forward(x, w, b)
    dropout_score, dropout_cache = dropout_forward(affine_relu_score, dropout_param)
    cache = (affine_relu_cache, dropout_cache)
    final_score = np.copy(dropout_score)
    return final_score, cache

def affine_relu_dropout_backward(dout, cache):
    affine_relu_cache, dropout_cache = cache
    ddropout = dropout_backward(dout, dropout_cache)
    daffine_relu, dw, db = affine_relu_backward(ddropout, affine_relu_cache)
    dx = np.copy(daffine_relu)
    return dx, dw, db