import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
from pdb import set_trace as trace
class Channel_Attention(layers.Layer):
        """
        Channel Attention Layer
        """
        def __init__(self, filters, reduction=1, name = None):

                super(Channel_Attention, self).__init__()
                self.conv1=layers.Conv2D(filters // reduction, kernel_size = (1,1), name =
                        'conv2d-1', activation='relu', padding  = 'same')
                self.conv2=layers.Conv2D(filters, kernel_size = (1,1), name =
                        'conv2d-2', activation='sigmoid', padding = 'same')
        def call(self, x):
                skip_conn=tf.identity(x,name = 'identity')
                channel=x.get_shape()[-1]
                x=tf.reshape(tf.reduce_mean(input_tensor = x, axis = [1,2]),
                                (-1,1,1,channel))
                x=self.conv1(x)
                x=self.conv2(x)
                x=tf.multiply(skip_conn,x)
                return x
class Scalar_CA(layers.Layer):
        """
        Channel Attention Layer, Returns vector of scalars
        """
        def __init__(self, filters, reduction=1, name = None):

                super(Scalar_CA, self).__init__()
                self.conv1=layers.Conv2D(filters // reduction, kernel_size = (1,1), name =
                        'conv2d-1', activation='relu', padding  = 'same')
                self.conv2=layers.Conv2D(filters, kernel_size = (1,1), name =
                        'conv2d-1', activation='sigmoid', padding = 'same')
                self.dense1=layers.Dense(2)
        def call(self, x):
                skip_conn=tf.identity(x,name = 'identity')
                channel=x.get_shape()[-1]
                x=tf.reshape(tf.reduce_mean(input_tensor = x, axis = [1,2]),
                                (-1,1,1,channel))
                x=self.conv1(x)
                x=self.conv2(x)
                x=tf.reshape(x, (-1,channel))
                x=self.dense1(x)
                return x

class Scale_Attention(layers.Layer):
        def __init__(self, filters, reduction=2):
                super(Scale_Attention, self).__init__()
                self.local_attention=layers.Conv2D(1, kernel_size = (7), 
                                     name='merge2d-1', activation = 'linear',
                                     padding='same',
                                     strides=1)
               
                self.conv1=layers.Conv2D(filters // reduction, kernel_size = (1,1), name =
                        'conv2d-1', activation='linear', padding  = 'same')
                self.conv2=layers.Conv2D(filters, kernel_size = (1,1), name =
                        'conv2d-1', activation='sigmoid', padding = 'same')        
        def call(self, x):
                skip_conn=tf.identity(x,name = 'identity')
                a=self.local_attention(x)
                x=self.conv1(a)
                x=self.conv2(x)
                return tf.multiply(skip_conn,x) 
class SOCA(layers.Layer):
        """
        Second Order Channel Attention Layer
        Based on:
        https://github.com/daitao/SAN/edit/master/TrainCode/model/MPNCOV/python/MPNCOV.py
        """
        def __init__(self, filters, reduction=1, input_shape = (48,48)):
                super(SOCA, self).__init__()
                self.conv_du=models.Sequential()
                self.conv_du.add(layers.Conv2D(filters // reduction,
padding ='same', activation = 'relu', kernel_size = (3,3)))
                self.conv_du.add(layers.Conv2D(filters, padding =
'same', activation='sigmoid', kernel_size = (3,3)))
                h,w=input_shape[0], input_shape[1]
                h,w=min(80,h) , min( 80, w)
                self.crop=layers.experimental.preprocessing.CenterCrop(h,w, name=None)
        def normalizeCov(self, x, iterN):
                batchSize, c=tf.shape(x)[0], x.shape[-1]
                h,w=x.shape[1], x.shape[2]
                I3=3 * tf.eye(c,c, batch_shape = (batchSize,))
                normA =tf.multiply( (1/3), tf.reduce_sum(tf.multiply(x,I3), axis= [1,2]))
                A=x / tf.reshape(normA, (batchSize,1,1))
                Y=tf.zeros((batchSize, c,c))
                Z=tf.eye(c,c, batch_shape = [batchSize])

                ZY=0.5 * (I3 - A)
                Y=tf.matmul(A,ZY)
                Z=ZY

                for i in range(1, iterN -1):

                    ZY=0.5 * (I3 - tf.matmul(Z,Y))
                    Y=tf.matmul(Y, ZY)
                    Z=tf.matmul(ZY, Z)
                ZY=0.5 * tf.matmul(Y, I3 - tf.matmul(Z,Y))
                y=ZY * tf.math.sqrt(tf.reshape(normA, (batchSize,1,1)))
                y=tf.reshape(tf.reduce_mean(y, axis = -1), (batchSize,1,1,c))
                
                return self.conv_du(y) 
        def call(self, x):

                self.crop.target_height=min(48,x.shape[1])
                self.crop.target_width=min(48,x.shape[2])
                x_sub=self.crop(x)
                x_sub=tf.expand_dims(x_sub, axis = -1)
                
                h1 ,w1=200,200
                ### compute covariance matrix
                h,w, c=x_sub.shape[1], x_sub.shape[2] ,x_sub.shape[3]
                batch_size=tf.shape(x)[0]
                M=(h * w )
                x_sub=tf.reshape(x_sub, (batch_size, c,M))
                Minv= tf.cast(1/M, dtype = tf.float32)
                I_hat=tf.multiply(Minv, tf.eye(M) - tf.multiply(Minv, tf.ones((M,M))))
                cov=tf.matmul(tf.matmul(x_sub,I_hat), tf.transpose(x_sub, perm= [0,2,1]))

                y_cov=self.normalizeCov(cov, 5)
                return y_cov * x  


