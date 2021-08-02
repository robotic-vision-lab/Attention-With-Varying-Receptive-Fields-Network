import tensorflow as tf
import pdb
def ens_mse(x,y):
    #yhat = tf.expand_dims(tf.reduce_mean(y, axis = 1),-1)
    yhat = tf.reduce_mean(y, axis = 1)
    mse = tf.reduce_mean(tf.square(yhat - x))
    uncertainty = tf.reduce_mean(tf.math.reduce_std(y, axis = 1))
    return mse #+ uncertainty
    
