import tensorflow as tf
import numpy as np
import os
import pdb
from PIL import Image
from PIL import ImageOps
BUFFER_SIZE = 1000
BATCH_SIZE = 16
IMG_WIDTH = 120
IMG_HEIGHT = 120
CHANNEL = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

def resize(x, scale_factor = 2, y_shape = (120,120), testing = False,
image_size = None, all_scales = False):
        """ 
        Returns resized tensor
        """ 
        if testing: size = [image_size[0] // scale_factor,image_size[1] //
scale_factor] 
        else: size = [y_shape[0] //scale_factor, y_shape[1] //scale_factor] 
        y = x 
        x = tf.image.resize(x, size, method=tf.image.ResizeMethod.BICUBIC)
        y = tf.image.resize(y, [x.shape[0] * scale_factor, x.shape[1] *
scale_factor], method = tf.image.ResizeMethod.BICUBIC)
        if all_scales: 
            y = [tf.image.resize(y, [int(y.shape[0] * i/4), int(y.shape[1] *
i/4)], method = tf.image.ResizeMethod.BICUBIC, antialias = True) for i in [2,3,4]]
        return x, y
def saveImage(x,  filename, dst_dir):
  """ 
  Saves tensor to filename in dst_dir
  """ 
  if not os.path.exists(dst_dir): os.makedirs(dst_dir)
  filename = dst_dir + '/' + filename
  channels = x[-1]
  jpeg = tf.io.encode_jpeg(tf.cast(x, tf.uint8))
  tf.io.write_file(filename, jpeg)

def normalize(x):
        """ 
        Divides  input tensor by 255
        :returns normalized input tensor
        """ 
        x = tf.divide(tf.cast(x, dtype = tf.float32),tf.constant([255], dtype =
tf.float32))
        return x

def random_crop(x, y_shape, channel = 1):
        """ 
        :returns a randomly cropped tensor from x with shape of y_shape
        """ 
        stacked_image =x
        cropped_image = tf.image.random_crop( stacked_image, size=[
y_shape[0], y_shape[1], channel])

        return cropped_image
def brightness_adjust(x):
        """ 
        :returns tensor with brightness randomly adjusted
        """ 
        stacked_image = x
        adjusted_image = tf.image.random_brightness(stacked_image, 125)
        return adjusted_image
def random_hue(x):
        """ 
        Randomly adjusts hue
        :returns x with hue randomly adjusted
        """ 
        stacked_image = x
        hueImage = tf.image.random_hue(stacked_image,0.2)
        return hueImage
def gaussian_noise(x):
        """ 
        Adds Gaussian Noise
        :returns tensor with added Gaussian Noise
        """ 
        noise = tf.random.normal(tf.shape(x), mean = 0, stddev = 10,dtype
= tf.float32)
        x = tf.add(x,noise)
        return x
        
def process_images(filename, crop = False, noise = False, bright = False, hue =
False, scale_factor = 2, y_shape = (120,120), testing = False, channel = 1,
image_size = None, src_dir = None, num_outputs = 1, all_scales = False,
know_induce = False, car = False):
        x = tf.io.read_file(src_dir + '/'+ filename) 
        x = tf.cast(tf.image.decode_jpeg(x, channels = channel), dtype =
tf.float32)
        if not testing: x = random_crop(x, y_shape, channel)
        #if not testing: x,y = random_hue(x,y)
        x, y = resize(x, scale_factor,y_shape, 
                    testing = testing,
                     image_size  = image_size,
                    all_scales = all_scales)
        #x = brightness_adjust(x) 
        #x = gaussian_noise(x)
        if not all_scales:x,y = tf.clip_by_value(x,0,255), tf.clip_by_value(y,0,255) 
        else:
          x = tf.clip_by_value(x,0,255) 
          y = [tf.clip_by_value(ypart,0,255) for ypart in y]
        if not car:x= normalize(x)
        if num_outputs != 1:
            layer_name = 'block'
            if not all_scales: y = {layer_name + str(i): y for i in range(num_outputs)}
            else: y = {layer_name + str(i): y_i for i , y_i in enumerate(y)}
        if know_induce: 
            y = {'hr': y, 'tf_op_layer_RealDiv': x}
        if car: 
            tmp = y
            y = {'hr': y, 'low_r': x} 
            x = tmp
        return x, y
def evaluate_process_images(filename, crop = False, noise = False, bright = False, hue =
False, scale_factor = 2, y_shape = (120,120), testing = False, channel = 1,
image_size = None, src_dir = None, num_outputs = 1, all_scales = False,
know_induce = False, car = False, model = None, creating_results =
False,output_dir = None, challenge = False):
        x = tf.io.read_file(src_dir + '/'+ filename) 
        x = tf.cast(tf.image.decode_jpeg(x, channels = channel), dtype =
tf.float32)
        if not testing: x = random_crop(x, y_shape, channel)
        #if not testing: x,y = random_hue(x,y)
        x, y = resize(x, scale_factor,y_shape, 
                    testing = testing,
                     image_size  = image_size,
                    all_scales = all_scales)
        #x = brightness_adjust(x) 
        #x = gaussian_noise(x)
        if not all_scales:x,y = tf.clip_by_value(x,0,255), tf.clip_by_value(y,0,255) 
        else:
          x = tf.clip_by_value(x,0,255) 
          y = [tf.clip_by_value(ypart,0,255) for ypart in y]

        test_input = x
        if not car:x= normalize(x)
        if num_outputs != 1:
            layer_name = 'block'
            if not all_scales: y = {layer_name + str(i): y for i in range(num_outputs)}
            else: y = {layer_name + str(i): y_i for i , y_i in enumerate(y)}
        if not challenge:
            result = model(tf.reshape(x, (1,) + x.shape))
            result = tf.reshape(result, result.shape[1:])
        else: 
            test_input = tf.reshape(y, (1,) + y.shape)
            result = model(normalize(test_input))
            result = tf.reshape(result, result.shape[1:])
            test_input = tf.reshape(test_input, test_input.shape[1:])
        if creating_results:
            saveImage(test_input,filename, os.path.join(output_dir,str(
                     scale_factor), 'inputs'))
            saveImage(result, filename, os.path.join(output_dir,str(
                     scale_factor), 'outputs'))
        if know_induce: 
            y = {'hr': y, 'tf_op_layer_RealDiv': x}
        if car: 
            tmp = y
            y = {'hr': y, 'low_r': x} 
            x = tmp
        if challenge: return result, result
        return result, y
def evaluate_process_images_ens(filename, crop = False, noise = False, bright = False, hue =
False, scale_factor = 2, y_shape = (120,120), testing = False, channel = 1,
image_size = None, src_dir = None, num_outputs = 1, all_scales = False,
know_induce = False, car = False, model = None, creating_results =
False,output_dir = None, challenge = False):
        x = tf.io.read_file(src_dir + '/'+ filename) 
        x = tf.cast(tf.image.decode_jpeg(x, channels = channel), dtype =
tf.float32)
        if not testing: x = random_crop(x, y_shape, channel)
        #if not testing: x,y = random_hue(x,y)
        x, y = resize(x, scale_factor,y_shape, 
                    testing = testing,
                     image_size  = image_size,
                    all_scales = all_scales)
        #x = brightness_adjust(x) 
        #x = gaussian_noise(x)
        if not all_scales:x,y = tf.clip_by_value(x,0,255), tf.clip_by_value(y,0,255) 
        else:
          x = tf.clip_by_value(x,0,255) 
          y = [tf.clip_by_value(ypart,0,255) for ypart in y]

        test_input = x
        if not car:x= normalize(x)
        if num_outputs != 1:
            layer_name = 'block'
            if not all_scales: y = {layer_name + str(i): y for i in range(num_outputs)}
            else: y = {layer_name + str(i): y_i for i , y_i in enumerate(y)}
        if not challenge:
            result = model(tf.reshape(x, (1,) + x.shape))
            count = 0
            sqr = tf.squeeze(result,axis = 0)
            for i  in range(sqr.shape[0]):
                saveImage(sqr[i],str(i) + 'ens' +filename,
                        os.path.join(output_dir,str(scale_factor),'ens_out'))
                count = count + 1

            stdm = tf.math.reduce_std(result, axis =1 )
            stdm = tf.expand_dims(tf.squeeze(stdm *255/ tf.reduce_max(stdm)),-1)
            result = tf.reduce_mean(result,axis = 1)
            
        
            result = tf.reshape(result, result.shape[1:])
        else: 
            test_input = tf.reshape(y, (1,) + y.shape)
            result = model(normalize(test_input))
            result = tf.reshape(result, result.shape[1:])
            test_input = tf.reshape(test_input, test_input.shape[1:])
        if creating_results:
            saveImage(test_input,filename, os.path.join(output_dir,str(scale_factor), 'inputs'))
            saveImage(result, filename, os.path.join(output_dir,str(scale_factor), 'outputs'))
            saveImage(stdm, filename,os.path.join(output_dir, str(scale_factor),'vars'))
        if know_induce: 
            y = {'hr': y, 'tf_op_layer_RealDiv': x}
        if car: 
            tmp = y
            y = {'hr': y, 'low_r': x} 
            x = tmp
        if challenge: return result, result
        return result, y

def bayes_process_images(filename , crop = False, noise = False, bright = False,
hue = False, scale_factor = 2, y_shape = (120,120), testing = False, channel =
1, image_size = None, src_dir = None, num_outputs = 1, all_scales = False,
know_induce = False, car = False, models = None, creating_results = False,
output_dir = None):
  """
  This function does not do anything special other than getting a merged image
from a list of model ensembles. This can be done in parallel, but all of  the
models have to be small in order to fit in memory. This function is for use with
a model that doesn't have any parameters, and simply returns  metrics. In this
fashion, the ensemble is evaluated in a highly optimized manner. 
  """
  x = tf.io.read_file(src_dir + '/' + filename)
  x = tf.cast(tf.image.decode_jpeg(x, channels = channel), dtype = tf.float32)
  x,y = resize(x, scale_factor, y_shape, testing = testing, image_size =
image_size, all_scales = all_scales)
  #x = gaussian_noise(x)
  x = normalize(x)
  x = tf.reshape(x, (1,) + x.shape)
  results = []
  for model in models: 
    result = model(x)
    result = tf.reshape(result, result.shape[1:])
    results.append(result)
  mergedImage = tf.math.reduce_mean(results, axis = 0)
  if creating_results == True:
    variance_map = tf.math.reduce_variance(results, axis = 0)
    var_max = tf.math.reduce_max(variance_map)
    std_map = tf.math.reduce_std(results, axis = 0)
    std_max = tf.math.reduce_max(std_map)
    variance_map = variance_map * 255 / var_max
    std_map = std_map * 255 / std_max 
    saveImage(variance_map,  filename, os.path.join(output_dir,'var'))
    saveImage(std_map, filename, os.path.join(output_dir, 'std'))
    saveImage(mergedImage,filename, os.path.join(output_dir, 'merge'))

    
    
  return mergedImage, y
        
def create_dataset(src_dir, shuffle = False, crop = False, scale_factor = 2,
y_shape = (120,120), testing = False, channel  = 1, num_outputs = 1, all_scales
= False, batch_size = 8, know_induce = False, car = False):
        filenames =os.listdir(src_dir) 
        images = []
        image_size = np.array(Image.open(os.path.join(src_dir,filenames[0]))).shape[:2] 
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda filename: process_images(filename, crop = crop, y_shape = 
                        y_shape, scale_factor = scale_factor, testing = testing,
                        channel = channel, image_size = image_size, src_dir =
                        src_dir, num_outputs = num_outputs, all_scales
                        =all_scales, know_induce = know_induce, car = car),
                      num_parallel_calls=AUTOTUNE)
        if shuffle: ds = ds.shuffle(128 if channel == 3 else 1000)
        ds = ds.batch(batch_size)
        return ds.prefetch(buffer_size = AUTOTUNE)
def evaluate_create_dataset(src_dir, shuffle = False, crop = False, scale_factor = 2,
y_shape = (120,120), testing = False, channel  = 1, num_outputs = 1, all_scales
= False, batch_size = 8, know_induce = False, car = False, model = None, creating_results = False, output_dir = None):
        filenames =os.listdir(src_dir) 
        images = []
        image_size = np.array(Image.open(os.path.join(src_dir,filenames[0]))).shape[:2] 
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda filename: evaluate_process_images(filename, crop = crop, y_shape = 
                        y_shape, scale_factor = scale_factor, testing = testing,
                        channel = channel, image_size = image_size, src_dir =
                        src_dir, num_outputs = num_outputs, all_scales
                        =all_scales, know_induce = know_induce, car = car, model = model, creating_results = creating_results, output_dir = output_dir),
                      num_parallel_calls=AUTOTUNE)
        #if shuffle: ds = ds.shuffle(128 if channel == 3 else 1000)
        ds = ds.batch(batch_size)
        return ds.prefetch(buffer_size = AUTOTUNE)
def evaluate_create_dataset_ens(src_dir, shuffle = False, crop = False, scale_factor = 2,
y_shape = (120,120), testing = False, channel  = 1, num_outputs = 1, all_scales
= False, batch_size = 8, know_induce = False, car = False, model = None, creating_results = False, output_dir = None):
        filenames =os.listdir(src_dir) 
        images = []
        image_size = np.array(Image.open(os.path.join(src_dir,filenames[0]))).shape[:2] 
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda filename: evaluate_process_images_ens(filename, crop = crop, y_shape = 
                        y_shape, scale_factor = scale_factor, testing = testing,
                        channel = channel, image_size = image_size, src_dir =
                        src_dir, num_outputs = num_outputs, all_scales
                        =all_scales, know_induce = know_induce, car = car, model = model, creating_results = creating_results, output_dir = output_dir),
                      num_parallel_calls=AUTOTUNE)
        if shuffle: ds = ds.shuffle(128 if channel == 3 else 1000)
        ds = ds.batch(batch_size)
        return ds.prefetch(buffer_size = AUTOTUNE)
def bayes_create_dataset(src_dir, shuffle = False, crop  = False, scale_factor =
2, y_shape = (120,120), testing = False, channel = 1, num_outputs = 1,
all_scales = False, batch_size = 8, know_induce = False, car = False, models =
None, creating_results = False, output_dir = None):
  """
  This dataset loader is only for use during evaluation of bayesian learning
models. This function needs a list of initialized models that can fit entirely
in memory. 
  """

  filenames = os.listdir(src_dir) 
  images = []
  image_size = np.array(Image.open(os.path.join(src_dir, filenames[0]))).shape[:2]
  ds = tf.data.Dataset.from_tensor_slices(filenames)
  ds = ds.map(lambda filename: bayes_process_images(filename, crop = crop, y_shape =
y_shape, scale_factor = scale_factor, testing = testing, channel = channel,
image_size = image_size, src_dir = src_dir, num_outputs = num_outputs,
all_scales = all_scales, know_induce = know_induce, car = car, models = models,
output_dir = output_dir, creating_results = creating_results),
num_parallel_calls = AUTOTUNE)
  if shuffle: ds = ds.shuffle(128 if channel == 3 else 1000)
  ds = ds.batch(batch_size)
  return ds.prefetch(buffer_size = AUTOTUNE)
       
        
