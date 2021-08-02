import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pdb
class AnchorRegularizer():
    def __init__(self, strength, n_weights,noise, batch_size = 16,  ):
      self.strength = strength
      self.n_data = batch_size
      self.anch_var = 1 / n_weights
      self.noise = noise
      self.n_weights = n_weights

      self.weights_lambda = noise / self.anch_var
      #self.bias_weights = np.random.normal(loc = 0, scale = np.sqrt(self.anch_var),
#size = [n_bias])
      self.anchor_weights = None
      self.anchors = []


    def get_anchor_weights():
      return self.anchor_weights
    def get_strength():
      return self.stength

    def weights_anchor(self,x):
      x1 = tf.cast(tf.reshape(x, [-1]), tf.float64)
      with tf.init_scope():
        if self.anchor_weights is None:
           npanch = np.random.normal(loc = 0, scale =np.sqrt(self.anch_var), size = x1.shape)
           self.anchor_weights = tf.constant(npanch)
           self.anchors.append(npanch)
      return tf.math.reduce_sum(tf.square(x1 - self.anchor_weights)) * self.weights_lambda / self.n_data
    def bias_anchor(x):
      return tf.math.sum(tf.square(weights - self.bias_weights)) * self.weights_lambda / self.n_data
      
    def bias_anchor(weights):
      """
      This function can be passed to the bias  regularizer function of a dense
layer for the purposes of anchoring weights of a layer to a certain
distribution.
      :param weights: Tensor of shape [n_hidden]
      :regularization term for layer
      example:
      hidden_param = 64
      anch_reg = AnchorRegularizer(.01, hidden_param, 16)
      model.add(layers.Dense(hidden_param, bias_regularizer = anch_reg.bias_anchor))
      """
      return tf.sum(tf.square(weights - self.bias_weights)) * self.weights_lambda / n_data
def add_regularizer(model, strength, n_weights, noise, batch_size):
    for layer in model.layers:
        if layer.trainable == True:
            areg = AnchorRegularizer(strength,n_weights,noise,batch_size)
            layer.kernel_regularizer = areg.weights_anchor
            breg = AnchorRegularizer(strength,n_weights,noise,batch_size)
            layer.bias_regularizer = breg.weights_anchor
