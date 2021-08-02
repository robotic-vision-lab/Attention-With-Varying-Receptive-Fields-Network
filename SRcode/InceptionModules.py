import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
import pdb
class inception_module(layers.Layer):
        """
        Inception Module
        This module is made for image super-resolution where max pooling layers
        are known to be unnecessary. In addition instead of different sized
        kernels, this module instead uses different dilation rates.
        This module as with all other modules in this file may be used the Keras 
        sequential API.
        """
        def __init__(self, filters,reduction = 1, name = None):
                super(inception_module, self).__init__()
                self.mergea = layers.Conv2D(filters //2, kernel_size = (1,1), 
                                     name = 'merge2d-1', activation = 'linear',
                                     padding = 'same',
                                     strides = (1,1))
                self.mergeb = layers.Conv2D(filters //2, kernel_size = (3,3),
                                     name = 'merge2d-2', activation = 'linear', 
                                     padding = 'same', 
                                     strides = 1)
                self.mergec = layers.Conv2D(filters //2, kernel_size = (5,5),
                                     name =  'merge2d-3', activation = 'linear', 
                                     padding = 'same', 
                                     strides = 1)
                self.pool = layers.MaxPool2D(pool_size = (3,3), strides = 1,
                                             padding = 'same')
        def call(self, x):
                a = self.mergea(x)

                b = self.mergeb(x)

                c = self.mergec(x)
                
                d = self.pool(x)
                return tf.concat([a,b,c,d], axis = -1)
class incept_dilated(layers.Layer):
        """
        Inception Module with different dilation rates
        This module is made for image super-resolution where max pooling layers
        are known to be unnecessary. In addition instead of different sized
        kernels, this module instead uses different dilation rates.
        This module as with all other modules in this file may be used the Keras 
        sequential API.
        """
        def __init__(self, filters,reduction = 2, name = None):
                super(incept_dilated, self).__init__()
                self.mergea = layers.Conv2D(filters // reduction, kernel_size = (1,1), 
                                     name = 'merge2d-1', activation = 'relu',
                                     strides = (1,1), padding = 'same')
                self.mergeb = layers.Conv2D(filters //reduction, kernel_size
=(3,3),
                                     name = 'merge2d-2', activation = 'relu', 
                                     padding = 'same', dilation_rate = 1,
                                     strides = 1)
                self.mergec = layers.Conv2D(filters //reduction, kernel_size =
(5,5),
                                     name =  'merge2d-3', activation = 'relu', 
                                     padding = 'same', dilation_rate = 1,
                                     strides = 1)
                #self.pool = layers.MaxPool2D(pool_size = (3,3), strides = 1,padding = 'same')                
        def call(self, x):
                a= self.mergea(x)

                b = self.mergeb(x)

                c = self.mergec(x)
                
                #d = self.pool(x)
                
                

                return tf.concat([a,b,c], axis = -1)
                


