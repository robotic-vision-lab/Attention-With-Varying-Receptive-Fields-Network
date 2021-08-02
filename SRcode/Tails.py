mport tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
import pdb
class Tail(layers.Layer):
        """
        Tail 
        """
        def __init__(self, c, k = (3,3), padding = 'same', name = None):
                super(Tail, self).__init__(name = name)
                self.conv1 = layers.Conv2D(filters = c,kernel_size = k, 
                     padding = padding)

        def call(self,x):
                x =  self.conv1(x)
                x = tf.clip_by_value(x,0,255)
                return x
class Tail_Ens(layers.Layer):
        """
        Tail Ensemble
        """
        def __init__(self, c, num_models = 100, name = None):
                super(Tail_Ens, self).__init__(name = name)
                self.c = c
                self.num_models = num_models
                self.tail = Tail(c * num_models)
                #self.dropout = layers.Dropout(rate = .5)

        def call(self,x):
            c,num_in_ens = self.c,self.num_models
            tail = self.tail(x)
            #tail = self.dropout(tail)
            ensemble = tf.split(tail, num_in_ens, axis = -1)
            newens = []
            for member in ensemble:
                newspl = tf.split(member,c, axis = -1)
                newmem = [tf.expand_dims(tf.squeeze(newspl[i], axis =
                    -1), axis = -1) for i in range(c)]
                newmem = tf.expand_dims(tf.concat(newmem, axis = -1), axis = 1)
                newens.append(newmem)
            ensConcat = tf.concat(newens, axis = 1)
            return ensConcat
