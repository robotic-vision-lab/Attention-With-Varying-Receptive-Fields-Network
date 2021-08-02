import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
import pdb
from pdb import set_trace as trace
from ResidualBodies import *
from Tails import *
from Upsamplers import *
from ResidualGroups import *
from ResidualBlocks import *
from AttentionLayers import *
from FeatureExtractors import *
from Regularizers import *
class Model_Lists():
        """
        This class can be used to custom build models on the fly.
        example: 
from tensorflow.keras import datasets, layers, models, regularizers
from custom_layers import Model_Lists;ml = Model_Lists()
rcan = ml.models['rcan']
rcan = ml.get_model(*rcan)
        
        """
        def __init__(self):
                self.models = {}
                self.models['rcan_deconv_incept'] = self.rcan_with_deconvolutions_and_incept
                self.models['rcan'] = self.get_rcan_sequential
                self.models['rcan_regularized'] = self.get_rcan_regularized
                self.models['rcan_ensembled'] = self.get_rcan_ensembled
                self.models['rcan_deconv'] = self.get_deconvolution
                self.models['rcan_deconv_sa'] = self.get_deconvolution_and_SA
                self.models['rcan_sa'] = self.rcan_and_SA
                self.models['rcan_incept'] = self.get_rcan_sequential_with_incept
                self.models['rcan_incept_dilated'] = self.get_rcan_incept_dilated
                self.models['rcan_summed_features'] = self.get_rcan_summed_features
                self.models['rcan_all_scales'] = self.get_rcan_all_scales

                self.models['rcan_multi_supervision'] = self.get_rcan_multi_supervision
                self.models['rcan_all_scales_dilated'] = self.get_rcan_all_scales_dilated
                self.models['rcan_all_scales_shared'] = self.get_rcan_all_scales_shared
                self.models['rcan_all_scales_shared_edge'] = self.get_rcan_all_scales_shared_edge
                self.models['rcan_all_scales_shared_complete'] = self.get_rcan_all_scales_shared_complete
                self.models['rcan_soca_dense'] = self.get_rcan_soca_dense
                self.models['rcan_soca'] = self.get_rcan_soca
                self.models['rcan_dense_dilated'] = self.get_rcan_dense_dilated
                self.models['rcan_all_scales_dense_dilated'] = self.get_rcan_all_scales_dense_dilated
                self.models['rcan_all_DDSOCA'] = self.get_rcan_all_DDSOCA 
                self.models['rcan_DDSOCA'] = self.get_rcan_DDSOCA 
                self.models['rcan_DDSOCAG'] = self.get_rcan_DDSOCAG
                self.models['rcan_DDSOCA_know'] = self.get_rcan_DDSOCA_know
                self.models['rcan_DDSOCA_car'] = self.get_rcan_DDSOCA_car
                self.models['rcan_DDSOCA_super'] = self.get_rcan_DDSOCA_super
                self.models['rcan_DDSOCA_stable'] = self.get_rcan_DDSOCA_stable
                self.models['rcan_all_DDSOCA_stable'] = self.get_rcan_all_DDSOCA_stable
                self.models['rcan_DDSOCA_smooth'] = self.get_rcan_DDSOCA_smooth
                self.models['rcan_DDSOCA_smooth_incr'] = self.get_rcan_DDSOCA_smooth_incr
                self.models['rcan_KIK'] = self.get_rcan_KIK
                self.models['rcan_denoiser'] = self.get_rcan_denoiser
                self.models['rcan_DDSOCA_smooth_anchored'] = self.get_rcan_DDSOCA_smooth_anchored
                self.models['rcan_DDSOCA_smooth_ens'] = self.get_rcan_DDSOCA_smooth_ens
                self.models['rcan_DDSOCA_smooth_ens_res'] = self.get_rcan_DDSOCA_smooth_ens_res
                self.models['rcan_smooth_body'] = self.get_rcan_smooth_body
                self.models['rcan_smooth_ens'] =self.get_rcan_smooth_ens
                self.models['Incre_rec'] = self.get_Incre_rec
                self.models['rcan_ens'] = self.get_rcan_ens
                self.models['MDSR_smooth'] = self.get_MDSR_smooth
                self.models['MDSR'] = self.get_MDSR
                self.models['rcan_dd'] = self.get_rcan_dd
                self.models['rcan_dd_comp'] = self.get_rcan_dd_comp
        def get_models(self, models):
                """
                :param models: String for key to models dict
                :returns list of models indexed in model dict by models strings
                example: 
                modelnames = [ 'rcan','rcan']
                modelfuncs = ml.get_models(modelnames)

                """
                return [self.models[model] for model in models]
         
        def get_rcan_model(self,input_shape = [60,60,1], f = 64,
c = 1, scale = 2):
                my_params = [[input_shape],[], [f,scale],[c]]
                my_layers  = [layers.InputLayer,residual_channel_attention_network,
Up_Scale_Module,Tail]
                return  my_params, my_layers
        def get_rcan_model_no_input(self,f = 64, c = 1, scale = 2):
                my_params = [[], [f,scale],[c]]
                my_layers  = [residual_channel_attention_network, Up_Scale_Module,Tail]
                return  my_params, my_layers
        def get_rcan_sequential(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(residual_channel_attention_network(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_dd(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_dd(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_dd_comp(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_dd_comp(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_ens(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5, num_models = 30):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(residual_channel_attention_network(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail_Ens(c,num_models))
                return model
        def get_rcan_smooth_body(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_smooth_body(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_MDSR(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(MDSR(f, n_blocks = b))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_MDSR_smooth(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(MDSR_smooth(f, n_blocks = b))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_regularized(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_smooth_body(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model

        def get_rcan_denoiser(self, input_shape, f = 64, c = 1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(residual_channel_attention_network(f, n_blocks = b, n_groups = g))
                model.add(Tail(c))
                return model
        def get_rcan_KIK(self,input_shape, f = 64, c = 1,scale = 2, b =
5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(residual_channel_attention_network(f, n_blocks = b, n_groups = g))


                #model.add(multiKernelUpscale(1, c = f, scale = scale))

                model.add(Up_Scale_Module(f, scale))
                model.add(layers.Conv2D(filters = f, kernel_size = (1,1), padding = 'same'))

                model.add(Tail(c))
                return model

        def rcan_with_deconvolutions_and_incept(self,input_shape, f = 64, c =
1,scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_with_inception(f, n_blocks = b))
                model.add(layers.Conv2DTranspose(f, strides =
                                                scale, padding = 'same', 
                                                kernel_size = (3,3),
                                                activation = 'linear'))
                model.add(Tail(c))
                return  model

        def get_deconvolution(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(residual_channel_attention_network(f, n_blocks = b))
                model.add(layers.Conv2DTranspose(f * 4, strides =
scale, padding = 'same', kernel_size = (3,3), activation = 'linear'))
                model.add(Tail(c))
                return model
        def get_deconvolution_and_SA(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_with_Scale_Attention(f, n_blocks = b))
                model.add(layers.Conv2DTranspose(f * 4, strides =
scale, padding = 'same', kernel_size = (3,3), activation = 'linear'))
                model.add(Tail(c))
                return model
        def rcan_and_SA(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_with_Scale_Attention(f, n_blocks = b, n_groups =g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_soca_dense(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_dense_dilated(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_Dilated(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_Incre_rec(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding ='same', activation = 'relu'))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding ='same', activation = 'relu'))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding
='same', activation = 'relu', dilation_rate = 2))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding
='same', activation = 'relu',dilation_rate = 2))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding
='same', activation = 'relu',dilation_rate = 2))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding ='same', activation = 'relu', dilation_rate = 2))
                model.add(layers.Conv2D(filters = 128, kernel_size = 5, padding ='same', activation = 'relu', dilation_rate = 2))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model

        def get_rcan_soca(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_soca(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_DDSOCA(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_DDSOCA_smooth_incr(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_smooth_incr(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_DDSOCA_smooth(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_smooth(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model
        def get_rcan_DDSOCA_smooth_ens(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5,num_models = 50):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_smooth(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail_Ens(c,num_models))
                return model
        def get_rcan_DDSOCA_smooth_ens_res(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5,num_models = 50):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_smooth_res(f, n_blocks = b, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail_Ens(c,num_models))
                return model
        def get_rcan_DDSOCA_smooth_anchored(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_smooth(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))

                add_regularizer( model,.001, f, .0001, 4)
                return model



        def get_rcan_DDSOCA_stable(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCA_stable(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model

     

        def get_rcan_DDSOCAG(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_Dense_SOCAG(f, n_blocks = b, n_groups = g, input_shape = input_shape))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail(c))
                return model

        def get_rcan_DDSOCA_know(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                x = RCAN_Dense_SOCA(f, n_blocks = b, n_groups = g, input_shape = input_shape)(inputs)
                x = Up_Scale_Module(f, scale)(x)

                xhr = tf.image.resize(inputs, (inputs.shape[1] * scale, inputs.shape[2] * scale)) * 255
                x = tf.concat([x,xhr], axis = -1)
                hr = Tail(c, name = 'hr')(x)
                model = keras.Model(inputs = inputs, outputs = hr)
                return model
        def get_rcan_DDSOCA_super(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)

                hrlanczos5 = tf.image.resize(inputs, (inputs.shape[1] * scale, inputs.shape[2] * scale),method = 'lanczos5')
                x = RCAN_Dense_SOCA(f, n_blocks = b, n_groups = g, input_shape = input_shape)(hrlanczos5)
                hr = Tail(c, name = 'hr')(x)
                model = keras.Model(inputs = inputs, outputs = hr)
                return model

        def get_rcan_DDSOCA_car(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                lr = layers.Conv2D(f, padding = 'valid', strides = (scale,
scale), kernel_size = (3,3))(inputs)

                lr = layers.Conv2D(f, padding = 'valid', strides = (scale,
scale), kernel_size = (3,3))(inputs)

                lr = layers.Conv2D(f, padding = 'valid', strides = (scale,
scale), kernel_size = (3,3))(inputs)

                lr = layers.Conv2D(c, padding = 'valid', strides = (scale,
scale), kernel_size = (3,3), name = 'low_r')(inputs)

                x = RCAN_Dense_SOCA(f, n_blocks = b, n_groups = g, input_shape =
input_shape)(lr)
                x = Up_Scale_Module(f, scale)(x)
                hr = Tail(c, name = 'hr')(x)
                model = keras.Model(inputs = inputs, outputs = [hr,lr])
                return model

        def get_rcan_all_scales(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                x = residual_channel_attention_network(f, n_blocks = b, n_groups
= g)(inputs)
                upa = Up_Scale_Module(f, 2)(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = Up_Scale_Module(f,4)(x)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_ensembled(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, num_in_ens = 50):
                inputs = keras.Input(input_shape)
                x = residual_channel_attention_network(f, n_blocks = b, n_groups
= g)(inputs)
                up = Up_Scale_Module(f, scale)(x)

                tail = Tail(c * num_in_ens,)(up)
                ensemble = tf.split(tail, num_in_ens, axis = -1)
                newens = []
                for member in ensemble:
                    newspl = tf.split(member,c, axis = -1)
                    newmem = [tf.expand_dims(tf.reduce_mean(newspl[i], axis =
                        -1), axis = -1) for i in range(c)]
                    newmem = tf.expand_dims(tf.concat(newmem, axis = -1), axis = 1)
                    newens.append(newmem)
                ensConcat = tf.concat(newens, axis = 1)
                outputs = [ensConcat]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_smooth_ens2(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, num_in_ens = 50):
                inputs = keras.Input(input_shape)
                x = rcan_smooth_body(f, n_blocks = b, n_groups= g)(inputs)
                up = Up_Scale_Module(f, scale)(x)
                tail = Tail(c * num_in_ens,)(up)
                tail = layers.Dropout(rate = .3)(tail)
                ensemble = tf.split(tail, num_in_ens, axis = -1)
                newens = []
                for member in ensemble:
                    newspl = tf.split(member,c, axis = -1)
                    newmem = [tf.expand_dims(tf.reduce_mean(newspl[i], axis =
                        -1), axis = -1) for i in range(c)]
                    newmem = tf.expand_dims(tf.concat(newmem, axis = -1), axis = 1)
                    newens.append(newmem)
                ensConcat = tf.concat(newens, axis = 1)
                outputs = [ensConcat]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_smooth_ens(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5,num_models = 30):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_smooth_body(f, n_blocks = b, n_groups = g))
                model.add(Up_Scale_Module(f, scale))
                model.add(Tail_Ens(c,num_models))
                return model

        def get_rcan_all_scales_shared(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                x = residual_channel_attention_network(f, n_blocks = b, n_groups
= g)(inputs)
                up2 = Up_Scale_Module(f,2)
                upa = up2(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = up2(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_all_scales_shared_complete(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                x = rcan_dilated_all_shared(f, n_blocks = b, n_groups
= g)(inputs)
                up2 = Up_Scale_Module(f,2)
                upa = up2(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = up2(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model

        def get_rcan_all_scales_shared_edge(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                inputs = keras.Input(input_shape)
                input_w_edges = CombineEdgeDetection()(inputs)
                x = residual_channel_attention_network(f, n_blocks = b, n_groups
= g)(input_w_edges)
                up2 = Up_Scale_Module(f,2)
                upa = up2(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = up2(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model


        def get_rcan_all_scales_dilated(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, d = 2):
                inputs = keras.Input(input_shape)
                x = rcan_dilated(f, n_blocks = b, d = d, n_groups = g)(inputs)
                upa = Up_Scale_Module(f, 2)(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = Up_Scale_Module(f,2)(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_all_scales_dense_dilated(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, d = 2):
                inputs = keras.Input(input_shape)
                x = RCAN_Dense_Dilated(f, n_blocks = b,  n_groups = g)(inputs)
                upa = Up_Scale_Module(f, 2)(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = Up_Scale_Module(f,2)(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_all_DDSOCA(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, d = 2):
                inputs = keras.Input(input_shape)
                x = RCAN_Dense_SOCA(f, n_blocks = b,  n_groups = g, input_shape
= input_shape)(inputs)
                upa = Up_Scale_Module(f, 2)(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = Up_Scale_Module(f,2)(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model
        def get_rcan_all_DDSOCA_stable(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5, d = 2):
                inputs = keras.Input(input_shape)
                x = RCAN_Dense_SOCA_stable(f, n_blocks = b,  n_groups = g, input_shape
= input_shape)(inputs)
                upa = Up_Scale_Module(f, 2)(x)
                upb = Up_Scale_Module(f, 3)(x)
                upc = Up_Scale_Module(f,2)(upa)

                taila = Tail(c, name = 'block0')(upa)
                tailb = Tail(c, name = 'block1')(upb)
                tailc = Tail(c, name = 'block2')(upc)
                
                outputs = [taila, tailb, tailc]
                model = keras.Model(inputs = inputs, outputs = outputs)
                return model






        def get_rcan_summed_features(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(rcan_summed_features(f, n_blocks = b))
                model.add(Up_Scale_Module(c, scale))
                model.add(Tail(c))
                return model
        def get_rcan_multi_supervision(self,input_shape, f = 64, c =
                              1, scale = 2, b = 5, k = (3,3), g = 5):

                inputs = keras.Input(shape = input_shape)
                x = layers.Conv2D(f,k, padding = 'same')(inputs)
                x = layers.Conv2D(f,k, padding = 'same')(x)
                x = Up_Scale_Module(c,scale)(x)
                outputs = []
                layer_name = 'block'
                for i in range(b):
                   x = rcab_no_res(f)( x)
                   x = layers.Conv2D(c,k, padding = 'same', name = layer_name + str(i))(x)
                   

                   outputs.append(x)

                model = keras.Model(inputs = inputs, outputs = outputs)
                keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

                return model




        def get_rcan_incept_dilated(self,input_shape, f = 64, c =
                              1, scale = 2,  b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(RCAN_incept_dilated(f, n_blocks = b))
                model.add(Up_Scale_Module(c, scale))
                model.add(Tail(c))
                return model

        def Scale_Att_Net(self,input_shape, f = 64, c = 1, scale = 2, b = 5, g = 5):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(Scale_Attention_Network(f, reduction = 1, n_blocks = b))
                model.add(layers.Conv2DTranspose(f * 4, strides = scale, padding = 'same', kernel_size = (3,3), activation = 'linear'))

                model.add(Tail(c))
                return model





        def small_inception_sequential(self,input_shape, f = 64, c = 1,scale = 2):
                model = models.Sequential()
                model.add(layers.InputLayer(input_shape))
                model.add(inception_module(f))
                model.add(layers.Conv2DTranspose(c, strides = 2,
kernel_size = (3,3), padding = 'same'))
                return  model

        def get_rcan_sequential_with_incept(self,input_shape, f = 64, c = 1,scale = 2):
                my_params = [[input_shape],[], [f,scale],[c]]
                my_layers  = [layers.InputLayer,RCAN_with_inception, Up_Scale_Module,Tail]
                model = self.get_model(my_params,my_layers)
                model = self.get_sequential_from_layers(model)
                return  model
        def get_rcan_seq_no_input(self, f = 64, c = 1,scale = 2):
                my_params = [[], [f,scale],[c]]
                my_layers  = [residual_channel_attention_network, Up_Scale_Module,Tail]
                model = self.get_model(my_params,my_layers)
                model = models.Sequential(model)
                return  model
       

        def get_model_from_params_layers(self, params,layers):
                
                return [layer(*param) for param,layer in zip(params,layers)]
        def get_model(self,params,layers):
                return self.get_model_from_params_layers(params,layers)
        def get_sequential_from_layers(self, layers):
                model = models.Sequential()
                for layer in layers: model.add(layer)
                return model
        def get_sequential_model(self,model_name):
                layers = self.models[model_name]
                layers = self.get_model(*layers)
                model = models.Sequential(layers)
                return model

