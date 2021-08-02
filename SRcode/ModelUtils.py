import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
import os
from Metrics import *
from Losses import *

def get_checkpoint(c_dir):
    """
    :param c_dir: string value pointing to the directory holding the
checkpoints
    :returns latest checkpoint compatible with keras saver API
    """
    
    if not os.path.exists(c_dir): os.makedirs(c_dir)
    checkpoints = [c_dir+ name  for name in os.listdir(c_dir)]
    if checkpoints:
            return max(checkpoints, key=os.path.getctime)
    else: return False

def get_params(layer):     
    """
    This function returns the parameters for use when loading weights of one
    model into another
    :param : layer : Keras Layer object
    :returns list of numpy arrays for all of the layers in the model
    """
    return [param.numpy() for param in layer.variables]
def load_model(checkpoints, input_shape, channels):
    model = keras.models.load_model(checkpoints, custom_objects =
{psnr.__name__: psnr,ssim.__name__: ssim} )
    model.build((None,) + input_shape + (channels,))
    return model
def load_model_ens(checkpoints, input_shape, channels):
    model = keras.models.load_model(checkpoints, custom_objects =
            {psnr_ens.__name__: psnr_ens,ssim_ens.__name__: ssim, ens_mse.__name__:ens_mse} )
    model.build((None,) + input_shape + (channels,))
    return model
def initialize_from_model_with_layers(model1,model2_dir, layers = [0,-2,-1]):
    checkpoints = get_checkpoint(model2_dir)
    if checkpoints is not False:
        c_init = checkpoints
        m_init = models.load_model(c_init, custom_objects = {'psnr': psnr,
'ssim': ssim})
        for l in layers:
            model1.layers[l].set_weights(get_params(m_init.layers[l]))
        model_init = None
    return model1
def initialize_from_model_with_layers_ens(model1,model2_dir, layers = [0,-2,-1]):
    checkpoints = get_checkpoint(model2_dir)
    if checkpoints is not False:
        c_init = checkpoints
        m_init = models.load_model(c_init, custom_objects = {'psnr_ens':
            psnr_ens,
            'ssim_ens': ssim_ens, 'ens_mse':ens_mse})
        for l in layers:
            model1.layers[l].set_weights(get_params(m_init.layers[l]))
        model_init = None
    return model1
def initialize_from_model(model1, model2):
    layers = len(model1.layers)
    for l in range(layers):
        model1.layers[l].set_weights(get_params(model2.layers[l]))
    return model1
def getStrategy():
    if tf.config.list_physical_devices('gpu'):
        strategy = tf.distribute.MirroredStrategy()
    else:   # use default strategy
        strategy = tf.distribute.get_strategy() 
    return strategy
