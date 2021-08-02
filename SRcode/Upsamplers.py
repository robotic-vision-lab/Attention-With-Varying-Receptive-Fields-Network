import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
class Pixel_Shuffle(layers.Layer):
        """
        Pixel Shuffle 
        """
        def __init__(self, scaling_factor):
                super(Pixel_Shuffle, self).__init__()
                self.scaling_factor = scaling_factor
                                
        @tf.function
        def call(self, x):
                sf = self.scaling_factor

                _, h, w, c = x.get_shape()
                c //= sf ** 2
                

                x = tf.split(x, self.scaling_factor, axis=-1)
                x = tf.concat(x, 2)

                x = tf.reshape(x, (-1, h * self.scaling_factor, w * self.scaling_factor, c))
                return x
class Kernel_Upscaling(layers.Layer):
  def __init__(self,k = 3,c = 1, scale = 2):
    super(Kernel_Upscaling, self).__init__()
    self.kernel = layers.Dense(k *scale * k *scale * c)
    self.k = k
    self.c = c
    self.flat = layers.Flatten()
    self.s = scale
  def call(self,x):
    k,c ,scale= self.k, self.c, self.s
    N = tf.shape(x)[0]
    H,W,C = x.shape[1:]
    pad_bottom = (k - H % k) if H % k != 0 else 0
    pad_right = (k - W % k) if W % k != 0 else 0
    H_pad,W_pad = H + pad_bottom, W + pad_right
    groupsH = H_pad//k
    groupsW = W_pad//k
    pad = [[0, pad_bottom], [0, pad_right]]
    space_to_batch = tf.space_to_batch(x, paddings = pad, block_shape= (groupsH,groupsW))
    flat_space = self.flat(space_to_batch)
    upScaled = self.kernel(flat_space)
    upScaled = tf.reshape(upScaled, (N * groupsH * groupsW, scale * k, scale *
k,c) )
    pad = [[0, pad_bottom], [0, pad_right]]

    out = tf.batch_to_space(upScaled,crops = pad, block_shape = [groupsH,groupsW])
    return out
class multiKernelUpscale(layers.Layer):
  def __init__(self, k = 3, c = 1, scale = 2):
    super(multiKernelUpscale, self).__init__()
    self.k1 = Kernel_Upscaling(k,c,scale)
  def call(self,x): 
    d = self.k1(x)
    return d


    

    
class BiCubic(layers.Layer):
  def __init__(self, scaling_factor):
    super(BiCubic, self).__init__()
    self.sc = scaling_factor
  def call(self, x):
    shape = tf.shape(x)
    x = tf.image.resize(x, shape[0] + (shape[0] * self.sc, shape[1] * self.sc,
shape[2]))
    return x
class OuterProductExp(layers.Layer):
  def __init__(self, scaling_factor):
    super(OuterProductExp, self).__init__()
    self.sc = scaling_factor
    self.conv = layers.Conv2D(filters * scaling_factor, kernel_size = (3,3),
padding = 'same', activation = 'relu')
  
  def call(self, x):
    x = self.conv(x)
    h,w,c = x.shape[1:]
    
    xh = tf.reshape(x, tf.shape(x)[0] + (self.sc * h * 2*w, c/ self.sc)) 
    xw = tf.reshape(x, tf.shape(x)[0] + (self.sc * h * 2*w, c/self.sc))
    outerProd = tf.matmul(xh,tf.transpose(xw))
def upScale(U,D,V,maxEigen, im_size, scaling_factor):
  approxImage = tf.zeros((1, im_size[0] * 2, im_size[1] * 2, im_size[2]))
  i = 0
def interpolate(u, scalingFactor):
  newv = tf.zeros((scalingFactor,1))
  j = 0
  for i in range(len(u)):
    newv[i] = u[i //2 -1] + (u[i //2 +1] - u[i //2 -1])/2
      
    
     
     
    
class Up_Scale_Module(layers.Layer):
        """
        Upscaling Module
        """
        def __init__(self,filters, scaling_factor):
                super(Up_Scale_Module, self).__init__()
                self.scaling_factor = scaling_factor
                if scaling_factor == 3:
                        self.conv1 = layers.Conv2D(filters * 9,kernel_size=
(1,1), padding = 'same') 
                        self.pixel_shuffle = Pixel_Shuffle(scaling_factor)
                elif scaling_factor & (scaling_factor -1) == 0:
                        self.log_scale_factor = int(np.log2(scaling_factor))
                        self.conv_layers = []
                        self.pixel_shuffle_layers = []
                        for i in range(self.log_scale_factor):
                                conv_layer = layers.Conv2D(filters * 4,kernel_size= (1,1),
                                        padding = 'same')
                                pixel_shuffle_layer = Pixel_Shuffle(2)
                                self.conv_layers.append(conv_layer)
                                self.pixel_shuffle_layers.append(pixel_shuffle_layer)
                else: raise NotImplementedError(" Not Supported Scale Factor %d", scaling_factor)
        @tf.function
        def call(self, x):
                if self.scaling_factor == 3:
                        x = self.conv1(x)
                        x = self.pixel_shuffle(x)
                elif self.scaling_factor & (self.scaling_factor -1) == 0:
                        for conv,pix in zip(self.conv_layers, self.pixel_shuffle_layers):
                                x = conv(x)
                                x = pix(x)
                return x

