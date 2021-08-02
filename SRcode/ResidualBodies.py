import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
from ResidualGroups import *
from pdb import set_trace as trace
BLOCKS =8 
GROUPS = 4 
class MDSR(layers.Layer):
        "MDSR"
        def __init__(self, f = 64, k = (3,3), n_blocks = BLOCKS):
          super(MDSR, self).__init__()
          self.head = layers.Conv2D(f,k, padding = 'same')
          self.rgb_mean = tf.reshape(tf.constant([0.4488, 0.4371, 0.4040]),[1,1,1,3])
          rgb_std = (1.0, 1.0, 1.0)
          self.pre_process = [res_block(f) for i in range(2)]
          self.body = [res_block(f) for i in range(n_blocks)]
          
        def call(self,x):
          x = x - self.rgb_mean
          x = self.head(x)
          for i in range(len(self.pre_process)):
            x = self.pre_process[i](x)
          res = self.body[0](x)
          for i in range(len(self.body[1:])):
            res = self.body[i](res)
          #x = x + self.rgb_mean
          return res
class MDSR_smooth(layers.Layer):
        "MDSR Smooth"
        def __init__(self, f = 64, k = (3,3), n_blocks = BLOCKS):
          super(MDSR_smooth, self).__init__()
          self.head = layers.Conv2D(f,k, padding = 'same')
          self.rgb_mean = tf.reshape(tf.constant([0.4488, 0.4371, 0.4040]),[1,1,1,3])
          rgb_std = (1.0, 1.0, 1.0)
          self.pre_process = [edsr_smooth_block(f) for i in range(2)]
          self.body = [edsr_smooth_block(f) for i in range(n_blocks)]
          
        def call(self,x):
          x = x - self.rgb_mean
          x = self.head(x)
          for i in range(len(self.pre_process)):
            x = self.pre_process[i](x)
          res = self.body[0](x)
          for i in range(len(self.body[1:])):
            res = self.body[i](res)
          #x = x + self.rgb_mean
          return res
class MSRN(layers.Layer):
        "MSRN"
        def __init__(self, f = 64, k = (3,3), n_blocks = BLOCKS):
          super(EDSR, self).__init__()
          self.head = layers.Conv2D(f,k, padding = 'same')
          rgb_mean = (0.4488, 0.4371, 0.4040)
          rgb_std = (1.0, 1.0, 1.0)
          self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
          self.add_mean = MeanShift(255,rgb_mean,rgb_std,1)
          self.body = [msrn_block(f) for i in range(n_blocks)]
          
        def call(x):
          x = self.sub_mean(x)
          x = self.head(x)
          res = x
          for i in range(len(self.body)):
            res = self.body[i](res)
          res += x
          x =  self.add_mean(x)

          return x

class EDSR(layers.Layer):
        "EDSR"
        def __init__(self, f = 64, k = (3,3), n_blocks = BLOCKS):
          super(EDSR, self).__init__()
          self.head = layers.Conv2D(f,k, padding = 'same')
          rgb_mean = (0.4488, 0.4371, 0.4040)
          rgb_std = (1.0, 1.0, 1.0)
          self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
          self.add_mean = MeanShift(255,rgb_mean,rgb_std,1)
          self.body = [res_block(f) for i in range(n_blocks)]
          
        def call(x):
          x = self.sub_mean(x)
          x = self.head(x)
          res = x
          for i in range(len(self.body)):
            res = self.body[i](res)
          res += x
          x =  self.add_mean(x)

          return x
class rcan_smooth_body(layers.Layer):
        """
        Residual Channel Attention Block with Smoothed Convolutions
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(rcan_smooth_body, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [smooth_group(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x

          
class residual_channel_attention_network(layers.Layer):
        """
        Residual Channel Attention Block
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(residual_channel_attention_network, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [residual_group(f, n_blocks = n_blocks,
name = 'group' + str(i)) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class RCAN_Dense_SOCA(layers.Layer):
        """
        Residual Channel Attention Block with SOCA
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks = BLOCK
                     S, input_shape = (48,48)):

                super(RCAN_Dense_SOCA, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [Dense_RG_SOCA(f,n_blocks = n_blocks, 
                      input_shape = input_shape) for i in range(n_groups)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class RCAN_Dense_SOCA_smooth(layers.Layer):
        """
        Residual Channel Attention Block with SOCA, Smoothed Convolutions
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks = 
                     BLOCKS, input_shape = (48,48)):

                super(RCAN_Dense_SOCA_smooth, self).__init__()
                self.head = layers.Conv2D(f,k, activation = 'relu', padding = 'same')
                self.body = layers.Conv2D(f,k, activation  = 'relu',padding = 'same')
                self.residual_groups = [Dense_RG_SOCA_smooth(f,n_blocks = 
                     n_blocks, input_shape = input_shape) for i in range(n_groups)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class RCAN_Dense_SOCA_smooth_res(layers.Layer):
        """
        Residual Channel Attention Block with SOCA, Different Dilation Rates, 
                     Smoothed convolutions
        """
        def __init__(self, f = 64,k = (3,3) , n_blocks = BLOCKS, input_shape = (48,48)):

                super(RCAN_Dense_SOCA_smooth_res, self).__init__()
                self.head = layers.Conv2D(f,k, activation = 'linear', padding = 'same')
                self.body = layers.Conv2D(f,k, activation  = 'linear',padding = 'same')
                self.residual_blocks = [RCAB_DD_Smooth_SOCA(f,input_shape = 
                     input_shape) for i in range(n_blocks)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for block in self.residual_blocks: x = block(x)
                x = self.body(x) + head 
                return x
class RCAN_Dense_SOCA_smooth_incr(layers.Layer):
        """
        Residual Channel Attention Block with SOCA, Smoothed Dilated Convolutions, Two-level Aggregation
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks = 
                     BLOCKS, input_shape = (48,48)):

                super(RCAN_Dense_SOCA_smooth_incr, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [Dense_RG_SOCA_smooth_incr(f,n_blocks = 
                     n_blocks, input_shape = input_shape) for i in range(n_groups)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x


class RCAN_Dense_SOCA_stable(layers.Layer):
        """
        Residual Channel Attention Block, with SOCA, dilated convolutions, 1 by
1 kernels
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks = BLOCKS, input_shape = (48,48)):

                super(RCAN_Dense_SOCA_stable, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [Dense_RG_SOCA_stable(f,n_blocks = 
                          n_blocks, input_shape = input_shape) for i in range(n_groups)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x

class RCAN_Dense_SOCAG(layers.Layer):
        """
        Residual Channel Attention Block, Geometric merging
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks = 
                     BLOCKS, input_shape = (48,48)):

                super(RCAN_Dense_SOCAG, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [Dense_RG_SOCAG(f,n_blocks = n_blocks, 
                     input_shape = input_shape) for i in range(n_groups)]                 

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x

class RCAN_Dense_Dilated(layers.Layer):
        """
        Residual Channel Attention Block, Dilated Convolutions
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(RCAN_Dense_Dilated, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [Dense_RG_dilated(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x


class rcan_dilated(layers.Layer):
        """
        Residual Channel Attention Bloc, dilated Convolutions
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS, d = 1):

                super(rcan_dilated, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [RG_dilated(f, n_blocks =
n_blocks, d = d) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class rcan_dilated_all_shared(layers.Layer):
        """
        Residual Channel Attention Block, DD
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS, d = 1):

                super(rcan_dilated_all_shared, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_group  = RG_dilated(f, n_blocks =n_blocks, d = d)
                self.n_groups = n_groups
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for i in range(self.n_groups): x = self.residual_group(x)
                x = self.body(x) + head 
                return x

class rcan_summed_features(layers.Layer):
        """
        Residual Channel Attention Block, summed groups
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(rcan_summed_features, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same', activation = 'relu')
                self.residual_groups = [RG_features_summed(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                features_summed = head
                for group in self.residual_groups: head +=  group(x)
                return head 
class rcan_multi_supervision(layers.Layer):
        """
        Residual Channel Attention Block, multiple supervision
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS, c = 1):

                super(rcan_multi_supervision, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.res_blocks = [residual_channel_attention_block(f) for i in
                                range(n_blocks)]

                self.tails = [layers.Conv2D(c,k,padding = 'same') for i in range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                outputs = []
                for block, tail in zip(self.res_blocks, self.tails):
                   x = block(x)
                   x = tail(x)
                   outputs.append(x)
                return outputs
class RCAN_with_inception(layers.Layer):
        """
        Residual Channel Attention Block with Inception
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(RCAN_with_inception, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [residual_group_with_inception(f,
n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class RCAN_incept_dilated(layers.Layer):
        """
        Residual Channel Attention Block with Inception modules
        that have different dilation rates
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(RCAN_incept_dilated, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [res_group_incept_dilated(f,
n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class RCAN_with_Scale_Attention(layers.Layer):
        """
        pdb.set_trace()
        Residual Channel Attention Block with Scale Attention
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(RCAN_with_Scale_Attention, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [residual_group_with_Scale_Attention(f,
n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x


class rcan_soca(layers.Layer):
        """
        Residual Channel Attention Block
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(rcan_soca, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [RG_SOCA(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class rcan_dd(layers.Layer):
        """
        Residual Channel Attention Block
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(rcan_dd, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [RG_dd(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x
class rcan_dd_comp(layers.Layer):
        """
        Residual Channel Attention Block, Dilated Convolutions
        """
        def __init__(self, f = 64,k = (3,3), n_groups = GROUPS, n_blocks =
BLOCKS):

                super(rcan_dd_comp, self).__init__()
                self.head = layers.Conv2D(f,k, padding = 'same')
                self.body = layers.Conv2D(f,k, padding = 'same')
                self.residual_groups = [RG_dd(f, n_blocks = n_blocks) for i in
                                range(n_groups)]
                

        def call(self, x):
                head = self.head(x) 
                x = head
                for group in self.residual_groups: x = group(x)
                x = self.body(x) + head 
                return x

