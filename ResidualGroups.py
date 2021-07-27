import tensorflow as tf
from tensorflow.keras import layers, models
from ResidualBlocks import *
class residual_group(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(residual_group, self).__init__()
                self.blocks = [residual_channel_attention_block(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class RG_dd(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(RG_dd, self).__init__()
                self.blocks = [rcab_dd(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class RG_dd_comp(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(RG_dd_comp, self).__init__()
                self.blocks = [rcab_dd_comp(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class smooth_group(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(smooth_group, self).__init__()
                self.blocks = [edsr_smooth_block(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class RG_dilated(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5, d = 5):

                super(RG_dilated, self).__init__()
                self.blocks = [rcab_dilated(filters, d = d) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same',
dilation_rate = d)
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x

class RG_features_summed(layers.Layer):
        """
        Residual Group
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(RG_features_summed, self).__init__()
                self.blocks = [rcab_features_summed(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'relu', padding  = 'same')
                
        def call(self, x):
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                return x

class residual_group_with_Scale_Attention(layers.Layer):
        """
        Residual Group With Scale Attention
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(residual_group_with_Scale_Attention, self).__init__()
                self.blocks = [RCAB_with_Scale_Attention(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class dense_res_group(layers.Layer):
        """
        Residual Group With Scale Attention
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(dense_res_group, self).__init__()
                self.blocks = [RCAB_with_Scale_Attention(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class residual_group_with_inception(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(residual_group_with_inception, self).__init__()
                self.blocks = [RCAB_with_inception(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class Dense_RG_SOCA(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5,
                     input_shape = (48,48)):

                super(Dense_RG_SOCA, self).__init__()
                self.blocks = [RCAB_dense_dilated_SOCA(filters, input_shape = input_shape) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class Dense_RG_SOCA_smooth(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5,
                     input_shape = (48,48)):

                super(Dense_RG_SOCA_smooth, self).__init__()
                self.blocks = [RCAB_DD_Smooth_SOCA(filters, input_shape = input_shape) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'relu', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class Dense_RG_SOCA_smooth_incr(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5,
                     input_shape = (48,48)):

                super(Dense_RG_SOCA_smooth_incr, self).__init__()
                self.blocks = [RCAB_DD_Smooth_SOCA_incr(filters, input_shape = input_shape) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'relu', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x



class Dense_RG_SOCA_stable(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5,
                     input_shape = (48,48)):

                super(Dense_RG_SOCA_stable, self).__init__()
                self.blocks = [RCAB_dense_dilated_SOCA_stable(filters, input_shape = input_shape) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x


class Dense_RG_SOCAG(layers.Layer):
        """
        Residual Group With Inception, geometric merging
        """
        def __init__(self, filters, name = None, n_blocks = 5,
                     input_shape = (48,48)):

                super(Dense_RG_SOCAG, self).__init__()
                self.blocks = [RCAB_dense_dilated_SOCAG(filters, input_shape = input_shape) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class Dense_RG_dilated(layers.Layer):
        """
        Residual Group With Inception
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(Dense_RG_dilated, self).__init__()
                self.blocks = [RCAB_dense_dilated(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x



class res_group_incept_dilated(layers.Layer):
        """
        Residual Group With Inception modules that apply different dilation
rates
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(res_group_incept_dilated, self).__init__()
                self.blocks = [RCAB_incept_dilated(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x
class RG_SOCA(layers.Layer):
        """
        Residual Group With Inception with SOCA
        """
        def __init__(self, filters, name = None, n_blocks = 5):

                super(RG_SOCA, self).__init__()
                self.blocks = [rcab_soca(filters) for i in range(n_blocks)]
                self.conv1 = layers.Conv2D(filters,kernel_size= (3,3), name =
                        'conv2d-1', activation = 'linear', padding  = 'same')
                
        def call(self, x):
                skip_conn = tf.identity(x,name = 'identity')
                for block in self.blocks: x = block(x) 
                x = self.conv1(x)
                x = x + skip_conn
                return x



