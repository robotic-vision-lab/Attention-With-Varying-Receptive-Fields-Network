from ModelUtils import *
from tfPreProcess import *
from Metrics import *
from  pdb import set_trace as trace
import pickle
from ImageUtils import getRawImage
from Losses import ens_mse
import csv
def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
def train(work_dir, scalen, restore, patch_shape, channel, filters, blocks,
groups, scale_init, dataset, histories, train_data, epochs,
val_data,  modelname, modelfunc, train_dir):

  checkpoint_dir = os.path.join(work_dir, str(scalen)+ '/')
  save_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir +
'checkpoint' , monitor = 'val_loss', save_best_only = True)
  lrate = LearningRateScheduler(step_decay)
  strategy = getStrategy()
  c_dir = os.path.join(work_dir, str(scalen) + '/')
  checkpoints = get_checkpoint(c_dir) 
  with strategy.scope():
    if checkpoints and restore == True: 

          model =load_model(checkpoints,patch_shape,channel)
    else:
        model = modelfunc(patch_shape +(channel,),
                           f = filters, c = channel, 
                          scale =  scalen, b= blocks, g = groups)            
        if scale_init is not None and not universal:
            ## initialize with weights from other scale
            c_init_dir = os.path.join(work_dir, str(scale_init) + '/')
            model = initialize_from_model_with_layers(model,c_init_dir,[0,-1,])
    model.summary()
    model.compile(loss = 'mse', optimizer = 'adam', metrics   =[psnr,ssim])
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    histories[modelname][dataset][scalen]['num_parameters'] = model.count_params()
  ####################################
  ## Train 
    #if is_training:
    history = model.fit(train_data,  epochs = epochs, 
                        validation_data= val_data, 
                        callbacks = [save_callback,lrate])
    #histories[dataset][scale]['history'] =history.history
    
    #history_filename =modelname  + os.path.basename(train_dir) + str(scalen) + 'history.pkl'
    with open('./Data/train_' + 'scale' + modelname + '.csv', 'a', newline = '\n') as f:
      writer = csv.writer(f)
      his = history.history
      epoch = 0
      for nloss,npsnr,nssim,v_l,v_p,v_s in zip(his['loss'],his['psnr'],his['
                     ssim'],his['val_loss'],his['val_psnr'],his['val_ssim']): 
        writer.writerow([dataset,scalen,filters,model.count_params(),epoch,
                     nloss,npsnr,nssim, v_l,v_p,v_s])
        epoch +=1
      
      
    #pdb.set_trace()
    #Save training history
    #with open(history_filename, 'wb') as f: pickle.dump(history.history , f) 
    del model; tf.keras.backend.clear_session()
def train_ens(work_dir, scalen, restore, patch_shape, channel, filters, blocks,
groups, scale_init, dataset, histories, train_data, epochs,
val_data,  modelname, modelfunc, train_dir):

  
  checkpoint_dir = os.path.join(work_dir, str(scalen)+ '/')
  save_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir +
'checkpoint' , monitor = 'val_loss', save_best_only = True)
  strategy = getStrategy()
  c_dir = os.path.join(work_dir, str(scalen) + '/')
  checkpoints = get_checkpoint(c_dir) 
  with strategy.scope():
    if checkpoints and restore == True: 

          model =load_model_ens(checkpoints,patch_shape,channel)
    else:
        model = modelfunc(patch_shape +(channel,),
                           f = filters, c = channel, 
                          scale =  scalen, b= blocks, g = groups)            
        if scale_init is not None and not universal:
            ## initialize with weights from other scale
            c_init_dir = os.path.join(work_dir, str(scale_init) + '/')
            model = initialize_from_model_with_layers_ens(model,c_init_dir,[0,-1,])
    model.summary()
    model.compile(loss = ens_mse, optimizer = 'adam', metrics
  =[psnr_ens,ssim_ens])
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    histories[modelname][dataset][scalen]['num_parameters'] = model.count_params()
  ####################################
  ## Train 
    #if is_training:
    history = model.fit(train_data,  epochs = epochs, 
                        validation_data= val_data, 
                        callbacks = [save_callback])
    #histories[dataset][scale]['history'] =history.history

    history_filename =modelname  + os.path.basename(train_dir) + str(scalen) + 'history.pkl'
    model = None
    #Save training history
    with open(history_filename, 'wb') as f: pickle.dump(history.history , f) 
    del model;tf.keras.backend.clear_session()

def train_Bayes(work_dir, scalen, restore, patch_shape, channel, filters, blocks,
groups, scale_init, dataset, histories, train_data, epochs,
val_data,  modelname, modelfunc, train_dir, num_bayes_models):
  """
  This function trains in the case that we are using an ensemble of models for
predictions. In this case, training should take place in a tf.function for loop
so that tensorflow limits the amount of tracing that takes place behind the
scenes. By the end of this function, each model should be ready for evaluation
and merging of detections.
  """
  print("Training Ensemble of models")
  for model_num in range(num_bayes_models):
    print(model_num)
    modelname2 = modelname + "_bayes_" + str(model_num) 
    checkpoint_dir = os.path.join(work_dir, "model" + str(model_num) + "scale" 
                     +  str(scalen)+ '/')
    save_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir +
  'checkpoint' , monitor = 'val_loss', save_best_only = True)
    strategy = getStrategy()
    c_dir = os.path.join(work_dir, str(scalen) + '/')
    checkpoints = get_checkpoint(c_dir) 
    with strategy.scope():
      if checkpoints and restore == True: 

            model =load_model(checkpoints,patch_shape,channel)
      else:
          model = modelfunc(patch_shape +(channel,),
                             f = filters, c = channel, 
                            scale =  scalen, b= blocks, g = groups)            
          if scale_init is not None and not universal:
              ## initialize with weights from other scale
              c_init_dir = os.path.join(work_dir, str(scale_init) + '/')
              model = initialize_from_model_with_layers(model,c_init_dir,[0,-1,])
      model.summary()
      model.compile(loss = 'mse', optimizer = 'adam', metrics
    =[psnr,ssim])
      keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
      histories[modelname][dataset][scalen]['num_parameters'] = model.count_params()
    ####################################
    ## Train 
      #if is_training:
      history = model.fit(train_data,  epochs = epochs, 
                          validation_data= val_data, 
                          callbacks = [save_callback])
      #histories[dataset][scale]['history'] =history.history

      history_filename =modelname2 + os.path.basename(train_dir) + str(scalen) + 'history.pkl'
      #Save training history
      with open(history_filename, 'wb') as f: pickle.dump(history.history , f) 
      tf.keras.backend.clear_session()
def evaluate_ens( test_data, test_dir, work_dir, filters, channel, scalen,
blocks, groups, histories, modelfunc, y_shape, batch_size, dataset, creating_results, modelname, all_scales):
  for t_dat, t_dir in zip(test_data, test_dir): 
      strategy = getStrategy()
      output_dir = os.path.join('Test_Save', dataset, os.path.basename(t_dir))
      in_shape = getRawImage(scalen, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:]
      out_shape = getRawImage(1, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:] 
      out_shape = (in_shape[0] * scalen, in_shape[1] * scalen, in_shape[2])
      with strategy.scope():
          #newmodel = modelfunc(t_dat._flat_shapes[0][1:], f= filters, 
                     #c = channel, scale = scale, b = blocks, g = groups)            
          #model = initialize_from_model(newmodel,model) 
          c_dir = os.path.join(work_dir, str(scalen) + '/')
          checkpoints = get_checkpoint(c_dir) 
          newmodel = modelfunc(in_shape, f= filters, 
                     c = channel, scale = scalen, b = blocks, g = groups)  
          model  =load_model_ens(checkpoints,in_shape,channel)
          model = initialize_from_model(newmodel, model)
          newmodel = None
          model.compile(loss = ens_mse, optimizer = 'adam',
metrics =[psnr_ens,ssim_ens])
          if all_scales:evaluation =  model.evaluate(t_dat)
          else: 
              #lossn,  psnrn, ssimn  = model.evaluate(t_dat)
              t_dat2 = create_dataset(t_dir, shuffle = True, scale_factor = scalen, y_shape = out_shape, channel = channel, batch_size = batch_size)
              t_dat = evaluate_create_dataset_ens(t_dir, shuffle = True, scale_factor = scalen, y_shape = out_shape, channel = channel, batch_size = batch_size, model = model, creating_results = creating_results, output_dir = output_dir) 
              metricModel = getMetricModel(out_shape)
              metricModel.compile(loss = 'mse', optimizer = 'adam', metrics =
                      [psnr,ssim])

              lossn, psnrn, ssimn = metricModel.evaluate(t_dat)
          del model;del t_dat; del newmodel;

          tf.keras.backend.clear_session()
      #Save Data
      if all_scales:
          t_dict = {}
          for i, ids in zip([2,3,4], [(1,3,10),(2,3,10),(3,3,10)]):
            l,p,s = evaluation[ids[0]:ids[2]:ids[1]]
            t_dict['loss' + str(i)] = l
            t_dict['psrn' + str(i)] = p
            t_dict['ssim' + str(i)] = s
      else: t_dict = {'loss':str(lossn), 'ssim':str(ssimn), 'psnr': str(psnrn)}
      histories[modelname][dataset][scalen]['metrics'][os.path.basename(t_dir)] = t_dict

def evaluate( test_data, test_dir, work_dir, filters, channel, scalen,
blocks, groups, histories, modelfunc, y_shape, batch_size, dataset, 
                     creating_results, modelname, all_scales):
  for t_dat, t_dir in zip(test_data, test_dir): 
      strategy = getStrategy()
      output_dir = os.path.join('Test_Save', dataset, os.path.basename(t_dir))
      in_shape = getRawImage(scalen, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:]
      #out_shape = getRawImage(1, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:] 
      im = getRawImage(scalen, os.path.join(t_dir, os.listdir(t_dir)[0]))[0]
      im2 = np.array(Image.fromarray(im.reshape(80,106)).resize((100,120)))
      im2 = im2.reshape((1,100,120,1))
      out_shape = (in_shape[0] * scalen, in_shape[1] * scalen, channel)
      with strategy.scope():
          #newmodel = modelfunc(t_dat._flat_shapes[0][1:], f= filters, 
                     #c = channel, scale = scale, b = blocks, g = groups)            
          #model = initialize_from_model(newmodel,model) 
          c_dir = os.path.join(work_dir, str(scalen) + '/')
          checkpoints = get_checkpoint(c_dir) 
          newmodel = modelfunc(in_shape, f= filters, 
                     c = channel, scale = scalen, b = blocks, g = groups)  
          model  =load_model(checkpoints,in_shape,channel)
          model = initialize_from_model(newmodel, model)
          newmodel = None
          model.compile(loss = 'mse', optimizer = 'adam',
metrics =[psnr,ssim])
          if all_scales:evaluation =  model.evaluate(t_dat)
          else: 
              #lossn,  psnrn, ssimn  = model.evaluate(t_dat)
              t_dat2 = create_dataset(t_dir, shuffle = True, scale_factor = 
                     scalen, y_shape = out_shape, channel = channel, batch_size = batch_size)
              t_dat = evaluate_create_dataset(t_dir, shuffle = True, 
                     scale_factor = scalen, y_shape = out_shape, channel = 
                     channel, batch_size = batch_size, model = model, 
                     creating_results = creating_results, output_dir = output_dir) 
              #out_shape = (out_shape[0] * scalen, out_shape[1] * scalen, channel)
              metricModel = getMetricModel(out_shape)
              metricModel.compile(loss = 'mse', optimizer = 'adam', metrics = [psnr,ssim])

              lossn, psnrn, ssimn = metricModel.evaluate(t_dat)
          tf.keras.backend.clear_session()
      #Save Data
      if all_scales:
          t_dict = {}
          for i, ids in zip([2,3,4], [(1,3,10),(2,3,10),(3,3,10)]):
            l,p,s = evaluation[ids[0]:ids[2]:ids[1]]
            t_dict['loss' + str(i)] = l
            t_dict['psrn' + str(i)] = p
            t_dict['ssim' + str(i)] = s
      else: t_dict = {'loss':str(lossn), 'ssim':str(ssimn), 'psnr': str(psnrn)}
      histories[modelname][dataset][scalen]['metrics'][os.path.basename(t_dir)] = t_dict
def evaluate_any_shape( test_data, test_dir, work_dir, filters, channel, scalen,
blocks, groups, histories, modelfunc, y_shape, batch_size, dataset,
creating_results, modelname, all_scales, epoch):

  c_dir = os.path.join(work_dir, str(scalen) + '/')
  checkpoints = get_checkpoint(c_dir) 
  newmodel = modelfunc((100,100,channel), f= filters, 
             c = channel, scale = scalen, b = blocks, g = groups)  
  model  =load_model(checkpoints,(100,100,channel),channel)
  model = initialize_from_model(newmodel, model)
  newmodel = None
  model.compile(loss = 'mse', optimizer = 'adam', metrics =[psnr,ssim])
  for t_dat, t_dir in zip(test_data, test_dir): 

      losses,psnrs,ssims = [],[],[]
      for filename in os.listdir(t_dir):
        
        lr,hr = getRawImage(scalen,os.path.join(t_dir,filename))
        hr = hr.astype(np.float32)
        model._layers[0] = layers.InputLayer((lr.shape[1],lr.shape[2],channel))
        pred = model(lr) 
        lossn,psnrn,ssimn = mse(hr,pred),psnr(hr,pred), ssim(hr,pred)
        losses.append(lossn);psnrs.append(psnrn);ssims.append(ssimn);
      losses,ssims,psnrs= np.mean(losses),np.mean(ssims), np.mean(psnrs)
      with open('./Data/test' + 'scale' + modelname + '.csv', 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow([dataset,scalen,filters,model.count_params(),nloss,npsnr,nssim])

def evaluate_bayes( test_data, test_dir, work_dir, filters, channel, scalen,
blocks, groups, histories, num_bayes_models, modelfunc):
  """
  This function evaluated previously saved models according to their loss, psnr,
and ssim scores. This method merges the results from three different models
before evaluating their scores.  For this to work, the method must loop through
all of the predictions of the models, obtaining psnr scores. This score should
provide us with the variance of each problem. In addition, based on the variance
provided by the problem, we can visualize the uncertainty of areas of each pixel
in the image with uncertainty maps.
  """
  for model_num in range(num_bayes_models):
    for t_dat, t_dir in zip(test_data, test_dir): 
        strategy = getStrategy()
        modelname  = modelname + "_bayes_" + str(model_num) 
        with strategy.scope():
            #newmodel = modelfunc(t_dat._flat_shapes[0][1:], f= filters, 
                       #c = channel, scale = scale, b = blocks, g = groups)            
            #model = initialize_from_model(newmodel,model) 
            c_dir = os.path.join(work_dir, "model" + str(model_num) + "scale" +  str(scalen) + '/')
            checkpoints = get_checkpoint(c_dir) 
            newmodel = modelfunc(t_dat._flat_shapes[0][1:], f= filters, 
                       c = channel, scale = scalen, b = blocks, g = groups)  
            model  =load_model(checkpoints,t_dat._flat_shapes[0][1:],channel)
            model = initialize_from_model(newmodel, model)
            newmodel = None

            model.compile(loss = 'mse', optimizer = 'adam',
  metrics =[psnr,ssim])
            if all_scales:evaluation =  model.evaluate(t_dat)
            else: lossn,  psnrn, ssimn  = model.evaluate(t_dat)
            tf.keras.backend.clear_session()
        #Save Data
        if all_scales:
            t_dict = {}
            for i, ids in zip([2,3,4], [(1,3,10),(2,3,10),(3,3,10)]):
              l,p,s = evaluation[ids[0]:ids[2]:ids[1]]
              t_dict['loss' + str(i)] = l
              t_dict['psrn' + str(i)] = p
              t_dict['ssim' + str(i)] = s
        else: t_dict = {'loss':str(lossn), 'ssim':str(ssimn), 'psnr': str(psnrn)}
        histories[modelname][dataset][scalen]['metrics'][os.path.basename(t_dir)] = t_dict
def eval_bayes(test_data, test_dir, work_dir, filters, channel,scalen, blocks,
groups, histories, num_bayes_models, modelfunc, y_shape, batch_size, dataset, creating_results, modelname):
  print('Evaluating Merged Models')
  strategy = getStrategy()
  with strategy.scope():
    for t_dir in test_dir: 
      output_dir = os.path.join('Test_Save', dataset, os.path.basename(t_dir))
      in_shape = getRawImage(scalen, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:]
      out_shape = getRawImage(1, os.path.join(t_dir, os.listdir(t_dir)[0]))[0].shape[1:] + (1,)
      models = []
      for model_num in range(num_bayes_models):
        c_dir = os.path.join(work_dir, 'model' + str(model_num) + 'scale' + str(scalen) + '/')
        checkpoints = get_checkpoint( c_dir)
        newmodel = modelfunc(in_shape , f = filters, c = channel,
scale = scalen, b = blocks, g = groups)
        model = load_model(checkpoints, in_shape, channel)
        model = initialize_from_model(newmodel, model)
        newmodel = None
        model.compile(loss = 'mse', optimizer  = 'adam', metrics = [psnr, ssim])
        models.append(model)
      t_dat = bayes_create_dataset(t_dir, shuffle = True, scale_factor = scalen,
y_shape = out_shape, channel = channel, batch_size = batch_size, models =
models, creating_results = creating_results, output_dir = output_dir)
      metricModel = getMetricModel(out_shape) 
      metricModel.compile(loss = 'mse', optimizer = 'adam', metrics = [psnr,ssim])
      print(t_dat._flat_shapes)
      metricModel.summary()
      m_loss, m_psnr, m_ssim = metricModel.evaluate(t_dat)
      tf.keras.backend.clear_session()
      t_dict = {'loss':str(m_loss), 'ssim':str(m_psnr), 'psnr': str(m_ssim)}
      histories[modelname][dataset][scalen]['metrics'][os.path.basename(t_dir)] = t_dict

      
        
      
    





