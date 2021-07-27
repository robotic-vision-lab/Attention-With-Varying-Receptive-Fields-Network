import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers

from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from PIL import ImageOps, Image
import os
from pdb import set_trace as trace
from tfPreProcess import brightness_adjust
import cv2
def create_inputs(src_dir,dst_dir = None, scale = 2, save = False, channel = 1):
    """
    :param src_dir: String representing path to input directory
    :param dst_dir: string representing path to output directory
    :param scale: Scale to reduce the input sizes by if not equal to one
    :param boolean value, where True implies inputs will be saved to dst_dir
    :returns images and the filenames naming each image.
    """
    
    filenames =os.listdir(src_dir) 
    images = []
    new_dst_dir = dst_dir
    if not os.path.exists(new_dst_dir): os.makedirs(new_dst_dir)
    for filename in filenames:
        if channel == 3: im_array = np.array(Image.open(os.path.join(src_dir,filename)))
        else:im_array =  np.array(ImageOps.grayscale(Image.open(os.path.join(src_dir,filename))))
        #im_array = brightness_adjust(im_array).numpy()
        filepath = os.path.join(dst_dir, filename)
        if scale != 1: 
            imPil = Image.fromarray(im_array)
    
            width, height = imPil.width, imPil.height
            imPil = imPil.resize(( width //scale, height //scale))
            im_array = np.array(imPil)
        if save:
            imPil = Image.fromarray(im_array) 
            imPil.save(filepath)

        images.append(im_array) 
    return images, filenames
def centerCrop(im, w, h):
    """
    :param im : pil image to be cropped 
    :param w : scalar value width of  output image
    :param h : scalar value height of output image
    :returns a pil image cropped to (w,h)
    """
    o_w, o_h = im.size
    left = ( o_w - w) /2
    top = ( o_h - h) /2
    right = (o_w + w) /2 
    bottom = (o_h + h) /2
    return im.crop((left,top,right,bottom))
def getRawImage(scale,filename, random_crop = False):
    """
    This function provides images in the appropriate format for input into
    tensorflow. 
    :param scale: Scalar value to downscale input image for network
    :param filename: String path to an image which will be loaded 
    :returns a downscaled and a ground truth grayscale image to be evaluated
    example: 
    x,y = getRawImage(4, myfile.jpg)
    model.evaluate(x,y)
    
    """
    im = Image.open(filename)
    
    im, im2 = ImageOps.grayscale(im), ImageOps.grayscale(im)
    if random_crop:
      hr = np.array(ImageOps.grayscale(im))
      x = np.random.randint(0,hr.shape[0] + 1 - 48 * scale)
      y = np.random.randint(0,hr.shape[1] + 1 - 48 * scale)
      hr = hr[x:x + 48 * scale, y : y + 48 * scale]
    
      lr = cv2.resize(np.array(hr),(48 * scale // scale, 48 * scale //scale), cv2.INTER_CUBIC)
      hr = cv2.resize(hr, (lr.shape[1] * scale, lr.shape[0] * scale))
    else:
      height, width = im.height, im.width
      im = Image.fromarray(cv2.resize(np.array(im),(width // scale, height//scale), cv2.INTER_CUBIC))
      im2 = im2.resize((im.width * scale , im.height * scale ), Image.BICUBIC)
      lr, hr = np.asarray(im), np.asarray(im2)
    lr = lr.reshape( (1,) + lr.shape + (1,))
    noise = np.random.normal(0,np.sqrt(10), lr.shape)
    lr = np.clip(lr + noise, 0,255)
    hr = hr.reshape( (1,) + hr.shape + (1,))
    return lr/255, hr
def getRawImagergb(scale,filename, isConverting = False):
    """
    This function provides images in the appropriate format for input into
    tensorflow. 
    :param scale: Scalar value to downscale input image for network
    :param filename: String path to an image which will be loaded 
    :returns a downscaled and a ground truth grayscale image to be evaluated
    example: 
    x,y = getRawImage(4, myfile.jpg)
    model.evaluate(x,y)
    
    """
    im = Image.open(filename).convert('RGB')
    neww,newh = min(im.width,800), min(im.height, 800)
    im = im.resize((neww,newh), Image.BICUBIC)
    #im, im2 = ImageOps.grayscale(im), ImageOps.grayscale(im)
    height, width = im.height, im.width
    im1 = im.resize((width // scale, height//scale), Image.BICUBIC)
    im2 = im.resize((im1.width * scale , im1.height * scale ), Image.BICUBIC)
    im, im2 = np.asarray(im1), np.asarray(im2)
    if isConverting: im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)
    im = im.reshape( (1,) + im.shape )
    im2 = im2.reshape( (1,) + im2.shape )
    return im/255, im2
def get_pil_from_array(image, channels, image_shape):
    im = image.numpy().astype(np.uint8).reshape(image_shape + (channels,))
    if channels != 1: return Image.fromarray(im)
    else: return Image.fromarray(im.reshape(im.shape[:-1]))
def reconstruct( patch_structure, patches):
    """
    This image reconstructs an output image of patch_structure
    :param: patch_structure: tuple of format (num_patches_height,
    num_patches_width)
    :param: patches: list of numpy arrays 
    :returns reconstructed image of size (patch_structure[0] * patch_height,
    patch_structure[1] * patch_width)
    """
    horiz_patches = []
    for i in range(patch_structure[0]):
        my_params = [[], [filters,scaling_factor],[output_channels]]
        my_layers  = [residual_channel_attention_network, Up_Scale_Module,Tail]
        return  my_params, my_layers

        horiz = np.concatenate(patches[i][:], axis = 1)
        horiz_patches.append(horiz)
    vert = np.concatenate(horiz_patches, axis = 0)
    
    return vert
def create_results(images,model, dst_dir = None, channels = 1, scale =
2, x_p_s= (60,60), save = False, filenames = None):
    """
    :param images : list of numpy arrays 
    :param: model : keras model object which must take in images[0].shape as
input.  
    :param: channels: scalar integer representing number of channels in
images.
    :param scale : scalat Integer value representing how much images will be
upscaled
    :param: x_p_s : shape of the input patches
    :param: save: Boolean value where True means output images will be saved
to dst_dir
    :param filenames: if images are being saved, filenames are used for the
name of each file.
    :return : None
    """
 
           
    fileIndex = 0
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    
    for image in images: 
        image = image / 255
        image_tensor = tf.convert_to_tensor(image.reshape((1,) +
image.shape + (1,)))
        patches = tf.image.extract_patches(image_tensor, sizes =
[1,x_p_s[0],x_p_s[1],channels], strides =
[1,x_p_s[0],x_p_s[1],channels],rates = [1,1,1,1],padding = 'VALID')
        patch_structure =( patches.shape[1],patches.shape[2])
        num_patches = np.prod(patch_structure)
        patches = tf.cast(tf.reshape(patches, shape =
[num_patches,x_p_s[0],x_p_s[1],channels]),
dtype = tf.float32)
        patches_hat = model(patches)
        patches_hat = patches_hat.numpy().reshape(patch_structure +
(x_p_s[0] * scale,x_p_s[1] * scale))
        
        reconstructed_image = reconstruct(patch_structure, patches_hat) 
        im = get_pil_from_array(reconstructed_image)
        if save: im.save(os.path.join(dst_dir,filenames[fileIndex]))
        fileIndex +=1




