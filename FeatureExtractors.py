import tensorflow as tf
from tensorflow.keras import layers, models
from ResidualBlocks import *
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import RandomNormal
import six
import numpy as np
from pdb import set_trace as trace
import sys

class CombineEdgeDetection(layers.Layer):
        """
        This Layer detects edges from an input image of any depth
        """
        def __init__(self, name = None):

                super(CombineEdgeDetection, self).__init__(name = name)
                                
        def call(self, x):
                s_e = tf.image.sobel_edges(x) 
                s_e = tf.squeeze(s_e,axis = -2)
                return tf.concat([x,s_e], axis = -1)

def tf_repeat(a, repeats, axis=0):
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a
def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])

def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.math.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals
def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a
def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals
class ConvOffset2D(Conv2D):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2D layer in Keras
        """

        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x):
        """Return the deformed featured map"""
        x_shape = x.get_shape()
        offsets = super(ConvOffset2D, self).call(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = tf_batch_map_offsets(x, offsets)

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape
        Because this layer does only the deformation part
        """
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

class MemNet( layers.Layer):
  def __init__(self, in_c, c, num_mem_blocks = 2, n_resblocks = 2):
    super(MemNet, self).__init__()
    self.feature_ext = BN_RELU(c, in_c)
    self.recon = BN_RELU( c, in_c)
    self.memory =[MemoryBlock(c, n_resblocks, i + 1) for i in range( num_mem_blocks)]
  def call(self, x):
    skip = x
    x = self.feature_ext(x)
    ys = [x]
    for mem in self.memory:
      x = mem(x, ys)
    x = self.recon(x)
    x = x + skip
    return x
class MemoryBlock(layers.Layer):
  def __init__(self, c, n_resblock, n_memblock):
    super(MemoryBlock, self).__init__()
    self.recursive_unit = [ResidualBlock(c) for i in range( n_resblock)]
    self.gate = BN_RELU((n_resblock + n_memblock) * c, c , 1,1,0)
  def call(self,x,ys):
    xs  = []
    skip = x
    for layer in self.recursive_unit: 
      x = layer(x)
      xs.append(x)
    
    gate_out = self.gate( tf.concat(xs + ys, -1))
    return ys + [gate_out]
class ResidualBlock(layers.Layer):
    def __init__(self, c, k = 3, s = 1, p = 1):
      super(ResidualBlock, self).__init__()
      self.relu_conv1 = BN_RELU(c,c,k,s,p)
      self.relu_conv2 = BN_RELU(c,c,k,s,p)
    def call(self,x):
      skip = x
      x = self.relu_conv1(x)
      x = self.relu_conv2(x)
      x = x + skip
      return x
class BN_RELU(layers.Layer):
    def __init__(self, in_c, c, k = 3, s =1, p =1):
      super(BN_RELU, self).__init__()
      #self.BN = layers.BatchNormalization(-1)
      self.relu = layers.ReLU()
      self.conv = layers.Conv2D(c,kernel_size = (k,k), strides = (s,s), padding =
'same')
    def call(self,x):
      x = tf.identity(x)
      #x = self.BN(x )
      x = self.relu(x)
      x = self.conv(x)
      return x


class SmoothDilated (layers.Layer):
  def __init__(self,f,k,dilation_factor, name, activation):  
    super(SmoothDilated, self).__init__(name = name)
    self.dilation_factor = dilation_factor
    self.fix_w_size = dilation_factor * 2 -1
    self.fix_w = tf.zeros(shape = [self.fix_w_size, self.fix_w_size,1,1,1])
    self.mask = np.zeros([self.fix_w_size,self.fix_w_size,1,1,1])
    self.mask[dilation_factor -1][ dilation_factor -1][ 0][0][0] = 1
    self.conv = layers.Conv2D(filters = f, kernel_size = k, padding = 'same', dilation_rate = self.dilation_factor, activation = activation)

  def call(self,x):
    fix_w = tf.add(self.fix_w, self.mask)
    x = tf.expand_dims(x,-1)
    x = tf.nn.conv3d(x,fix_w, strides = [1,1,1,1,1], padding = 'SAME')
    x = tf.squeeze(x,-1)
    x = self.conv(x)
    return x
    
   
  
