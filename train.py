import tensorflow as tf
import random
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models, regularizers
from  tfPreProcess import create_dataset 

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
from Metrics import *
from ImageUtils import *
from dirNav import *
from TrainEvalUtils import *
import csv

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
           

def list_full_paths(directory):
  return [os.path.join(directory, file) for file in os.listdir(directory)]


if __name__ == '__main__':

    args = ast.literal_eval(str(sys.argv))
    scales = [int(args[1])]
    patch_shape = (48,48)
    channels = [1]
    filters = 64
    blocks = 4
    groups = 2
    batch_size = 8
    # Each of the following lists will be for one iteration of 
    # outer testing loop. There may be more than one testing_directory
    # for in each of the test_dirs tuples
30
    saved_dirs = 'saved_dirs.json'
    datasets = [ 'thermal']
    dirTypes = ['train_dirs', 'test_dirs','val_dirs']
    save_dirs = False # This opens a GUI to make dataset selection easy
    if save_dirs: select_dirs(datasets,dirTypes,saved_dirs)
    dir_dict = json.load(open(saved_dirs,'r'))
    json_filename = 'histories.json'
    creating_results = True # For use when creating sample results
    is_evaluating = True# For use when evaluating the model
    is_training = True# For use when training a model
    save = True
    restore = False # For use to restore or start model from scratch
    multi_supervision = False # For use when multiple loss functions being used.
    know_induce = False # For use when ensuring that output downscales to input
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
    epochs = 300  
    histories = {} 
  
    modelnames = [ 'rcan_DDSOCA']
    opt = keras.optimizers.Adam(.0001) # Beginning learning rate
    modelfuncs = ml.get_models(modelnames)
    for modelfunc, modelname in zip(modelfuncs, modelnames):
      
      histories[modelname] = {}
      for dataset, channel in zip(datasets, channels): 
          train_dir = dir_dict[dataset]['train_dirs'][0]
          test_dir = dir_dict[dataset]['test_dirs']
          val_dir = dir_dict[dataset]['val_dirs'][0] 
          histories[modelname][dataset] = {}
          scale_init = None
          for scale in scales: 
              ################################################################
              ### CREATE HEADER OF TESTING AND TRAINING RESULTS
              ################################################################
              with open('./Data/train_'+ dataset + str(scale) + modelname + '.csv', 'w', newline = '\n') as f:
                writer = csv.writer(f)
                writer.writerow(['Dataset','scale','Filters','Parameters','Epoch','Loss','PSNR','SSIM'])
              for t_dir in test_dir:
                with open('./Data/test' + os.path.basename(t_dir) + str(scale) + modelname + '.csv', 'w', newline = '\n') as f:
                  writer = csv.writer(f)
                  writer.writerow(['Dataset','Test Set','scale','Filters','Parameters','loss','psnr', 'ssim', 'lossstd', 'psnrstd', 'ssimstd'])
              ################################################################
              ### COMPILE MODEL AND CREATE WORKING DIRECTORY IF IT DOES NOT EXIST
              ################################################################
              work_dir = os.path.join(modelname, dataset, str(scale))
              if not os.path.exists(work_dir): 
                os.makedirs(work_dir) 
              c_dir = os.path.join(work_dir, str(scale) + '/')
              model = modelfunc((100 ,100,channel), f= filters, c = channel, scale = scale, b = blocks, g = groups)  
              model.compile(loss = 'mse', optimizer = 'adam', metrics =[psnr,ssim])
              model.summary()
              
              if restore: model.load_weights(os.path.join(work_dir,
'checkpoint/checkpoint') )
              epoch = 1
              best_loss = 100000
              for epoch in range(epochs):
                print('epoch ', epoch, '/', epochs)
                idx = 0
                train_data = {i: getRawImage(scale,os.path.join(train_dir,filename),True)for i,filename in enumerate(os.listdir(train_dir))}
                train_indices = list(range(len(train_data)))
                for idx in range(len(train_data.keys())//batch_size):
                  ids = random.choices(train_indices, k = batch_size)
                  batchx,batchy = [train_data[idx][0] for idx in ids],[train_data[idx][1] for idx in ids]
                  lr ,hr= np.concatenate(batchx, axis = 0), np.concatenate(batchy,axis = 0)
                  hr = hr.astype(np.float32)
                  model._layers[0] = layers.InputLayer((lr.shape[1],lr.shape[2],channel))
                  with tf.GradientTape() as tape:
                    ##########################################################
                    ### GET MODEL PREDICTION
                    ##########################################################
                    pred = model(lr)
                    l1_reg_term = sum([tf.reduce_sum(tf.abs(_var)) for _var in model.trainable_variables])
                    loss = mse(hr,pred)# .0001 *l1_reg_term
                    psnrn = psnr(hr,pred)
                    ssimn = ssim(hr,pred)
                    grads = tape.gradient(loss,model.trainable_variables)
                  print('loss: ', loss.numpy(), 'ssim: ', ssimn.numpy(), 'psnr: ', psnrn.numpy())
                  if idx % 10 == 0: print('step: ', idx)
                  ############################################################
                  ### Apply Weight Updates
                  ############################################################
                  opt.apply_gradients(zip(grads,model.trainable_variables))
                  with open('./Data/train_' +dataset + str(scale) + modelname + '.csv', 'a', newline = '\n') as f:
                    writer = csv.writer(f)
                    writer.writerow([dataset,scale,filters,model.count_params(),epoch,loss.numpy(),psnrn.numpy(),ssimn.numpy()])
                  #if loss.numpy()< best_loss: 
                  #  best_loss = loss.numpy()
                  if epoch == epochs - 1 and save:
                    model.save_weights(os.path.join(work_dir, 'checkpoint/checkpoint'))
                  ############################################################
                  ## Evaluate
                  ############################################################
                  if is_evaluating and idx % 400  == 0  :
                    for t_dir in  test_dir: 
                        if t_dir == "KAIST":
                          tempfilenames = []
                          for k in kaistdirs: 
                            tempfilenames += list_full_paths(k);tempfilenames.sort()
                          filenames = []
                          for idx,t in enumerate(tempfilenames):
                            if idx % 200 == 0: filenames.append(t)
                          
                        else:
                          filenames = list_full_paths(t_dir);filenames.sort()
                        losses,psnrs,ssims = [],[],[]
                        for tidx,filename in enumerate(filenames,1):
                          
                          lr,hr = getRawImage(scale,filename)
                          hr = hr.astype(np.float32)
                          model._layers[0] = layers.InputLayer((lr.shape[1],lr.shape[2],channel))
                          pred = model(lr) 
                          lossn,psnrn,ssimn = mse(hr,pred),psnr(hr,pred), ssim(hr,pred)
                          losses.append(lossn);psnrs.append(psnrn);ssims.append(ssimn);
                          #####################################################
                          ###  CREATING RESULTS
                          #####################################################
                          if creating_results and epoch == epochs -1 or epoch == 0 :
                              pd =Image.fromarray(pred.numpy().squeeze().astype(np.uint8))
                              lrd = Image.fromarray((lr * 255).squeeze().astype(np.uint8))
                              if not os.path.exists(work_dir + '/' +os.path.basename(t_dir)+ '/predictions'):
                                os.makedirs(work_dir + '/' + os.path.basename(t_dir) + '/predictions')
                              if not os.path.exists(work_dir  + '/'+ os.path.basename(t_dir) + '/inputs'):
                                os.makedirs(work_dir + '/'+ os.path.basename(t_dir)+ '/inputs')
                              pd.save(os.path.join(work_dir,os.path.basename(t_dir) + '/predictions/image' + str(tidx) + '.png'))
                              lrd.save(os.path.join(work_dir,os.path.basename(t_dir)+ '/inputs/image' + str(tidx) + '.png'))
                        lossmean,ssimmean,psnrmean= np.mean(losses),np.mean(ssims), np.mean(psnrs)
                        lossstd,ssimstd,psnrstd = np.std(losses),np.std(ssims),np.std(psnrs)
                        ######################################################
                        ### WRITE DATA TO ./DATA/ DIRECTORY
                        ######################################################
                        with open('./Data/test'+os.path.basename(t_dir) + str(scale) + modelname + '.csv', 'a', newline = '\n') as f:
                          writer = csv.writer(f)
                          writer.writerow([dataset,os.path.basename(t_dir),scale,filters,model.count_params(),lossmean,psnrmean,ssimmean, lossstd,ssimstd,psnrstd])                 

              ################################################################
              ## Learning Rate Schedule
              ################################################################
                if epoch == 10: opt.lr.assign(1e-5)
                elif epoch == 60: opt.lr.assign(5e-6)
                elif epoch == 60: opt.lr.assign(2.5e-6)
                elif epoch == 90: opt.lr.assign(1e-6)

          tf.keras.backend.clear_session()
          scale_init = scale
          with open(json_filename, 'w') as f: json.dump(histories,f)
    print('\a')
    
    


