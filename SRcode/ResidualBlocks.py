import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
from AttentionLayers import *
from InceptionModules import *
from FeatureExtractors import *
class res_block(layers.Layer):
        """
        Residual  Block
        """
        def __init__(self, filters, name = None):

                super(res_block, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name 
                          = 'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'linear', padding = 'same')
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.conv2(x)
                x = x + skip_conn
                return x

class residual_channel_attention_block(layers.Layer):
        """
        Residual Channel Attention Block
        """
        def __init__(self, filters, name = None):

                super(residual_channel_attention_block, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'linear', padding = 'same')
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class rcab_dd_comp(layers.Layer):
        """
        Residual Channel Attention Block with Dilated Convolutions
        """
        def __init__(self, filters, name = None):

                super(rcan_dd_comp, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same', dilation_rate = 2)
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'linear', padding = 'same',
dilation_rate = 2)
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class MeanShift(layers.Conv2D):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign = -1):
        super(MeanShift, self).__init__(3,3,kernel_size = 1)
        std = tf.constant(rgb_std)
        self.weight.data

class rcab_dilated(layers.Layer):
        """
        Residual Channel Attention Block with Dilated Convolutions and Batch Normalization
        """
        def __init__(self, filters, name = None, d = 1):

                super(rcab_dilated, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same', dilation_rate = d)
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.bn1 = layers.BatchNormalization()
                self.bn2 = layers.BatchNormalization()
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x

class rcab_features_summed(layers.Layer):
        """
        Channel Attention Block (No Residual)
        """
        def __init__(self, filters, name = None):

                super(rcab_features_summed, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.bn1 = layers.BatchNormalization()
                self.bn2 = layers.BatchNormalization()
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.channel_attention(x)
                return x

class rcab_no_res(layers.Layer):
        """
        Residual Channel Attention Block
        """
        def __init__(self, filters, name = None):

                super(rcab_no_res, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.bn1 = layers.BatchNormalization()
                self.bn2 = layers.BatchNormalization()
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.channel_attention(x)
                return x
class RCAB_with_Scale_Attention(layers.Layer):
        """
        Residual Channel Attention Block With Scale Attention
        """
        def __init__(self, filters, name = None):

                super(RCAB_with_Scale_Attention, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.bn1 = layers.BatchNormalization()
                self.bn2 = layers.BatchNormalization()
                self.channel_attention = Scale_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)

                x = self.bn1(x)
                x = self.conv2(x)

                x = self.bn2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class RCAB_with_inception(layers.Layer):
        """
        RCAB with Inception, Different Kernel Sizes
        """
        def __init__(self, filters, name = None):

                super(RCAB_with_inception, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.incept = inception_module(filters)
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.incept(x)
                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class RCAB_incept_dilated(layers.Layer):
        """
        RCAB with Inception, Different dilation rates
        """
        def __init__(self, filters, name = None):

                super(RCAB_incept_dilated, self).__init__()
                self.incept = incept_dilated(filters)
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')

                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                x = self.conv1(x)
                x = self.incept(x)

                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class RCAB_dense_dilated(layers.Layer):
        """
        RCAB with Inception, Different dilation rates
        """
        def __init__(self, filters, name = None, d= 1):

                super(RCAB_dense_dilated, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name =
'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'relu', padding = 'same',
                        dilation_rate = 1)
                self.conv3 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-3', activation = 'relu', padding = 'same',
                        dilation_rate = 1)
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-4', activation = 'linear', padding = 'same')




                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                a = self.conv1(x)
                
                b = self.conv2(x)
                c  = self.conv3(x)
                attent = tf.concat([a,b,c], axis  = -1 )
                
                d = self.conv4(attent)
                
                
                x = self.channel_attention(d)
                x = x + skip_conn
                return x
class RCAB_dense_dilated_SOCA(layers.Layer):
        """
        RCAB with Inception, Different dilation rates
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(RCAB_dense_dilated_SOCA, self).__init__()

                
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name =
'conv2d-1', activation = 'relu', padding  = 'same')

                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'relu', padding = 'same',
dilation_rate = 2)
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-4', activation = 'linear', padding = 'same')
                
                self.channel_attention = SOCA(filters, input_shape = input_shape)
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                a = self.conv1(x)
                
                b = self.conv2(x)
                #c  = self.conv3(x)
                attent = tf.concat([a,b], axis  = -1 )              
                #attent = (a + b +c)/3
                d = self.conv4(attent)
                x = self.channel_attention(d)
                x = x + skip_conn
                return x
class rcab_dd(layers.Layer):
        """
        RCAB with Inception, Different dilation rates
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(rcab_dd, self).__init__()

                
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name =
'conv2d-1', activation = 'relu', padding  = 'same')

                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'relu', padding = 'same',
dilation_rate = 2)
                self.conv3 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-3', activation = 'relu', padding = 'same')
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-4', activation = 'linear', padding = 'same')
                
                #self.channel_attention = SOCA(filters, input_shape = input_shape)
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                a = self.conv1(x)
                
                b = self.conv2(x)
                #c  = self.conv3(x)
                attent = tf.concat([a,b], axis  = -1 )                
                #attent = (a + b +c)/3
                d = self.conv4(attent)
                #x = self.channel_attention(d)
                x = d + skip_conn
                return x

class RCAB_DD_Smooth_SOCA(layers.Layer):
        """
        RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(RCAB_DD_Smooth_SOCA, self).__init__()
                self.filters = filters

                
                self.conv1 = SmoothDilated(filters, k = (3,3), name ='conv2d-1',
                      activation = 'relu', dilation_factor = 1)
                self.conv2 = SmoothDilated(filters,k= (5,5), name ='conv2d-3',
activation = 'relu', dilation_factor = 1)
                self.conv3 = SmoothDilated(filters,k= (7,7), name ='conv2d-3',
activation = 'relu', dilation_factor = 1)
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =   
                     'conv2d-4', activation = 'relu', padding = 'same')

                #self.bn1 = layers.BatchNormalization()
                #self.bn2 = layers.BatchNormalization()
                #self.bn3 = layers.BatchNormalization()
                self.channel_attention = Channel_Attention(filters)
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                x = self.conv1(x) 
                #a = self.bn1(a)
                x = self.conv2(x)
                #b = self.bn2(b)
                x = self.conv3(x)
                #c = self.bn3(c)
                x = self.conv4(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class edsr_smooth_block(layers.Layer):
        """
        RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(edsr_smooth_block, self).__init__()
                self.filters = filters

                
                self.conv1 = SmoothDilated(filters, k = (3,3), name ='conv2d-1',
activation = 'relu', dilation_factor = 2)
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =                         'conv2d-4', activation = 'linear', padding = 'same')

                self.channel_attention = Channel_Attention(filters)
                
        def call(self, x):

                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.conv4(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x
class RCAB_DD_Smooth_SOCA_incr(layers.Layer):
        """
        RCAB with Inception, Different dilation rates, SOCA, smoothed dilations
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(RCAB_DD_Smooth_SOCA_incr, self).__init__()
                self.filters = filters

                
                self.conv1 = SmoothDilated(filters, k = (3,3), name =
'conv2d-1', activation = 'relu', dilation_factor = 2)
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-2', activation = 'relu', padding = 'same')
                self.conv3 = SmoothDilated(filters,k= (3,3), name =
                        'conv2d-3', activation = 'relu', dilation_factor = 3)
                self.conv4 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-4', activation = 'linear', padding = 'same')
                self.conv5 = layers.Conv2D(filters,kernel_size = (3,3), name =
'conv2d-5', activation = 'relu', padding = 'same')
                self.channel_attention = SOCA(filters, input_shape = input_shape)
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')

                a = self.conv1(x) 
                b = self.conv2(a)
                c = self.conv3(b)
                attent = tf.concat([a,b,c], axis = -1) 
                x = self.conv4(attent)
                attent2 = tf.concat([x,c], axis = -1)
                x = self.conv5(attent2)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x



class RCAB_dense_dilated_SOCA_stable(layers.Layer):
        """
        RCAB with Inception, Different dilation rates
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(RCAB_dense_dilated_SOCA_stable, self).__init__()
                self.conv1 = layers.Conv2D(filters,kernel_size= (1,1), name =
                        'conv2d-1', activation = 'relu', padding = 'same',)
                self.conv2 = layers.Conv2D(filters,kernel_size= (1,1), name =
                        'conv2d-4', activation = 'linear', padding = 'same')
                self.channel_attention = SOCA(filters, input_shape = input_shape)

        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x  = self.conv1(x)
                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x

class RCAB_dense_dilated_SOCAG(layers.Layer):
        """
        RCAB with Inception, Different dilation rates, Geometric Averaging
        """
        def __init__(self, filters, name = None, d= 1, input_shape = (48,48)):

                super(RCAB_dense_dilated_SOCAG, self).__init__()
                self.deformedMap = ConvOffset2D(filters * 2, name = 'conv2d-1')

                self.deformedMap2 = ConvOffset2D(filters * 2, name = 'conv2d-2')
                self.conv = layers.Conv2D(filters,kernel_size= (3,3), name =
                       'conv2d-4', activation = 'relu', padding = 'same')

                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                       'conv2d-5', activation = 'linear', padding = 'same')
                #self.ca2 = Channel_Attention(filters)
                self.channel_attention = SOCA(filters, input_shape = input_shape)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                

                a = self.deformedMap(x)
                a = self.conv(a)
                a = self.deformedMap2(a)
                a = self.conv2(a)
                x = self.channel_attention(a)
                #x2 = self.ca2(a)
                x = x + skip_conn
                return x
class msrn_block(layers.Layer):
  """
  multi scale residual network
  """
  def __init__(self, filters, name = None):
    super(msrn_block, self).__init__()
    self.conva1 = layers.Conv2D(filters, kernel_size = (5,5), padding = 'same',
activation = 'relu')
    self.conva2 = layers.Conv2D(filters, kernel_size = (3,3,), padding = 'same',
activation = 'relu')
    self.convb1 = layers.Conv2D(filters, kernel_size = (5,5), padding = 'same',
activation = 'relu')
    self.convb2 = layers.Conv2D(filters, kernel_size = (3,3), padding = 'same',
activation = 'relu')
    self.convc = layers.Conv2D(filters, kenel_size = (1,1), padding = 'same')
  def call(self,x):
    a1 = self.conva1(x)
    skip_conn = tf.identity(x, name = 'identity')
    a2 = self.conva2(x)
    merge1 = tf.concat([a1,a1], axis = -1)
    b1 = self.convb1(merge1)
    b2 = self.convb1(merge1)
    merge2 = tf.concat([b1,b2], axis = -1)
    c = self.convc(merge2)
    return c + skip_conn
    
class rcab_soca(layers.Layer):
        """
        Residual Channel Attention Block with SOCA
        """
        def __init__(self, filters, name = None):

                super(rcab_soca, self).__init__()
                self.conv1 = layers.Conv2D(filters, kernel_size = (3,3), name = 
                     'conv2d-1', activation = 'relu', padding  = 'same')
                self.conv2 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding = 'same')
                self.channel_attention = Channel_Attention(filters)
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.channel_attention(x)
                x = x + skip_conn
                return x

