import tensorflow as tf
import numpy
import numpy as np

from math import log10, sqrt
from PIL import ImageOps, Image
import sys
import os
import ast
import json
from pdb import set_trace as trace
from srGUI import select_dirs
from ModelLists import *
import pickle
from ModelUtils import *
from numpy import c_
from Metrics import *
from ImageUtils import *
from dirNav import *
from TrainEvalUtils import *
import csv
import cv2
import random
from SSIM_PIL import compare_ssim
from scipy import signal
from scipy import ndimage

import gauss
def normalize(x):
  std = np.std(x) + .01
  x = x/std
  xsum = np.sum(x)
  x = x/xsum
  return x

def list_full_paths(directory):
  return [os.path.join(directory, file) for file in os.listdir(directory)]
def PSNR(original, compressed):
  mse = np.mean((original - compressed) ** 2)
  if(mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
      return 100
  max_pixel = 255.0
  psnr = 20 * log10(max_pixel / sqrt(mse))
  return psnr     


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
def MSE(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def SSIM(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

@tf.function(experimental_relax_shapes=True)
def predict(model,t):
    return model(t)

if __name__ == '__main__':

    args = ast.literal_eval(str(sys.argv))
    scales = [2,3,4]
    patch_shape = (48,48)
    channels = [1]
    filters = 64
    blocks =  6
    groups = 3
    batch_size = 6
    # Each of the following lists will be for one iteration of 
    # outer testing loop. There may be more than one testing_directory
    # for in each of the test_dirs tuples
    saved_dirs = 'saved_dirs.json'
    datasets = [ 'thermal']
    dirTypes = ['train_dirs', 'test_dirs','val_dirs']
    kaistdirs = ["./Datasets/set06/images/set06/V000/lwir/",
"./Datasets/set06/images/set06/V001/lwir/",
"./Datasets/set06/images/set06/V002/lwir/",
"./Datasets/set06/images/set06/V003/lwir/",
"./Datasets/set06/images/set06/V004/lwir/",
"./Datasets/set07/images/set07/V000/lwir/",
"./Datasets/set07/images/set07/V001/lwir/",
"./Datasets/set07/images/set07/V002/lwir/",
"./Datasets/set08/images/set08/V000/lwir/",
"./Datasets/set08/images/set08/V001/lwir/",
"./Datasets/set08/images/set08/V002/lwir/",
"./Datasets/set09/images/set09/V000/lwir/",
"./Datasets/set10/images/set10/V000/lwir/",
"./Datasets/set10/images/set10/V001/lwir/",
"./Datasets/set11/images/set11/V000/lwir/",
"./Datasets/set11/images/set11/V001/lwir/"]
    save_dirs = False # This opens a GUI to make dataset selection easy
    if save_dirs: select_dirs(datasets,dirTypes,saved_dirs)
    dir_dict = json.load(open(saved_dirs,'r'))
    json_filename = 'histories.json'
    creating_results = True # For use when creating sample results
    is_evaluating = True# For use when evaluating the model
    save = True
    is_training = True# For use when training a model
    restore = False # For use to restore or start model from scratch
    multi_supervision = False # For use when multiple loss functions being used.
    know_induce = False # For use when ensuring that output downscales to input
    car = False # For use when applying noise reduction at beginning layers
    universal = False # For use when applying same scale for all other scales
    bayes = False # True if using bayesian ensembling
    num_bayes_models = 5
    universalScale = 4
    num_outputs = blocks if multi_supervision else 1
    all_scales = False
    if all_scales == True: scales = [4]
    ml = Model_Lists()

    tf.config.experimental_run_functions_eagerly(False)
    epochs = 120  
    histories = {} 
  
    opt = keras.optimizers.Adam(.0001)
    modelname = 'bicubic'
      
    for dataset, channel in zip(datasets, channels): 
        train_dir = dir_dict[dataset]['train_dirs'][0]
        test_dir = dir_dict[dataset]['test_dirs']
        val_dir = dir_dict[dataset]['val_dirs'][0] 
        scale_init = None
        for method in ['normal']:
          for scale in scales: 
              ################################################################
              ### CREATE HEADERS FOR DATA TO BE SAVED
              ################################################################
              with open('./Data/train_' + dataset + str(scale) + method + 
                     modelname + '.csv', 'w', newline = '\n') as f:
                writer = csv.writer(f)
                writer.writerow(['Dataset','scale','Filters','Parameters','
                     Epoch','Loss','PSNR','SSIM'])
              for t_dir in test_dir:
                with open('./Data/test' + os.path.basename(t_dir) + str(scale) 
                     + method + modelname + '.csv', 'w', newline = '\n') as f:
                  writer = csv.writer(f)
                  writer.writerow(['Dataset','Test Set','scale','Filters','
                     Parameters','loss','psnr', 'ssim', 'lossstd', 'psnrstd', 'ssimstd'])
              work_dir = os.path.join(modelname, dataset, str(scale))
              if not os.path.exists(work_dir): os.makedirs(work_dir) 
              c_dir = os.path.join(work_dir, str(scale) + '/')
              ################################################################
              ### EVALUATE FOR EACH TESTING DIRECTORY FOUND IN SAVED_DIRS.JSON
              ################################################################
              for t_dir in  test_dir: 

                  losses,psnrs,ssims = [],[],[]
                  if t_dir == "KAIST":
                    tempfilenames = []
                    for k in kaistdirs: 
                      tempfilenames += list_full_paths(k);
                    tempfilenames.sort()
                    filenames = []
                    for idx,t in enumerate(tempfilenames):
                      if idx % 200 == 0: filenames.append(t)
                    
                  else:
                    filenames = list_full_paths(t_dir);filenames.sort()
                  for tidx,filename in enumerate(filenames,1):
                    ##########################################################
                    ### CREATE PREDICTION USING BICUBIC INTERPOLATION
                    ##########################################################
                    lr,hr = getRawImage(scale,filename)
                    hr = hr.astype(np.float32)
                    sr = Image.fromarray((lr * 255).squeeze())
                    pred = np.expand_dims(np.array(sr.resize((sr.width * scale,
                     sr.height * scale))), axis = -1)
                    pred = np.expand_dims(pred, axis = 0)
                    loss = mse(hr,pred)
                    psnrn = psnr(hr,pred)
                    ssimn = ssim(hr,pred)
                    losses.append(loss );psnrs.append(psnrn);ssims.append(ssimn);
                    ##########################################################
                    ### SAVE DEGRADED INPUT IMAGES AND SUPER-RESOLUTION RESULTS 
                    ##########################################################
                    if creating_results :
                      pd =Image.fromarray(pred.squeeze().astype(np.uint8))
                      lrd = Image.fromarray((lr * 255).squeeze().astype(np.uint8))
                      if not os.path.exists(work_dir + '/' +os.path.basename(t_dir)+ '/predictions'):
                        os.makedirs(work_dir + '/' + os.path.basename(t_dir) + '/predictions')
                      if not os.path.exists(work_dir  + '/'+ os.path.basename(t_dir) + '/inputs'):
                        os.makedirs(work_dir + '/'+ os.path.basename(t_dir)+ '/inputs')
                      pd.save(os.path.join(work_dir,os.path.basename(t_dir) + '/predictions/image' + str(tidx) + '.png'))
                      lrd.save(os.path.join(work_dir,os.path.basename(t_dir)+ '/inputs/image' + str(tidx) + '.png'))
                  lossmean,ssimmean,psnrmean= np.mean(losses),np.mean(ssims), np.mean(psnrs)
                  lossstd,ssimstd,psnrstd = np.std(losses),np.std(ssims),np.std(psnrs)
                  ############################################################
                  ### STORE NEW DATA TO ./DATA/ DIRECTORY
                  ############################################################
                  with open('./Data/test'+os.path.basename(t_dir) +
str(scale) + method + modelname + '.csv', 'a', newline = '\n') as f:
                    writer = csv.writer(f)
                    writer.writerow([dataset,os.path.basename(t_dir),scale,
                     filters,0,lossmean,psnrmean,ssimmean, lossstd,ssimstd,psnrstd])                 
          restore = True 
        epochs = 5
        tf.keras.backend.clear_session()
        scale_init = scale
          
    print('\a')
    
    


