import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import models, transforms





def train_model(model, dataloaders, lossFunc,
                optimizer, scheduler, num_epochs=50,
                model_name= 'model', pgdFunc=None, nClasses = 100, work_dir='./', device='cpu', print_each = 1):


    trackRecords = {}
    trackRecords['weightNorm'] = []
    trackRecords['acc_test'] = []
    trackRecords['acc_train'] = []
    trackRecords['weights'] = []
    log_filename = os.path.join(work_dir, model_name+'_train.log')    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())    
    best_loss = float('inf')
    best_acc = 0.
    best_perClassAcc = 0.0
      
    for epoch in range(num_epochs):  
        if epoch%print_each==0:
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()


        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if epoch%print_each==0:
                print(phase)

            predList = np.array([])
            grndList = np.array([])
            
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode                
            else: 
                model.eval()   # Set model to evaluate mode
                
            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0
            countSmpl = 0.
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                image, segMask, label = sample
                
                image = image.to(device)
                segMask = segMask.to(device) 
                label = label.type(torch.long).view(-1).to(device)
                
                
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
##                loss = 0
                with torch.set_grad_enabled(phase=='train'):
##                    if phase=='train':  
##                        model.train()  # Set model to training mode                      
##                    else:
##                        model.eval()   # Set model to evaluate mode

                    
                    A = image * segMask   
                    
                    outputs = model(A)
                    
                    #print(image.shape, outputs.shape, label.shape)
                    
                    loss = lossFunc(outputs, label)

                    error = loss
                    logits = outputs
                    labelList = label
                    imageList = image
                    
                    softmaxScores = outputs.softmax(dim=1)
                    predLabel = softmaxScores.argmax(dim=1).detach().type(torch.float)                  
                    accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                    accRate = (accRate==0).type(torch.float).mean()

                    predList = np.concatenate((predList, predLabel.cpu().numpy()))
                    grndList = np.concatenate((grndList, labelList.cpu().numpy()))
                    grndList = np.round(grndList)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics  
                iterCount += 1
                sampleCount += label.size(0)
                running_acc += accRate*labelList.size(0) 
                running_loss_CE += error.item() * labelList.size(0) 
                running_loss = running_loss_CE

                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                
                       
##                del loss                
                
##                if iterCount%freqShow==0:
##                    print('\t{}/{} loss:{:.3f}'.
##                          format(iterCount, len(dataloaders[phase]), print2screen_avgLoss))
##                    fn = open(log_filename,'a')        
##                    fn.write('\t{}/{} loss:{:.3f}\n'.
##                             format( iterCount, len(dataloaders[phase]), print2screen_avgLoss))
##                    fn.close()
                    
                    
##            epoch_loss = print2screen_avgLoss
##                    
##            print('\tloss: {:.6f}'.format(epoch_loss))
##            fn = open(log_filename,'a')
##            fn.write('\tloss: {:.6f}\n'.format(epoch_loss))
##            fn.close()
##            
##            # deep copy the model            
##            path_to_save_paramOnly = os.path.join(work_dir, 'epoch-{}.paramOnly'.format(epoch+1))
##            torch.save(model.state_dict(), path_to_save_paramOnly)
##            
##            if (phase=='val' or phase=='test') and epoch_loss<best_loss:
##                best_loss = epoch_loss
##
##                path_to_save_paramOnly = os.path.join(work_dir, 'bestValModel.paramOnly')
##                torch.save(model.state_dict(), path_to_save_paramOnly)
##                
##                file_to_note_bestModel = os.path.join(work_dir,'note_bestModel.log')
##                fn = open(file_to_note_bestModel,'a')
##                fn.write('The best model is achieved at epoch-{}: loss{:.6f}.\n'.format(epoch+1,best_loss))
##                fn.close()
                    

            epoch_error = print2screen_avgLoss

##            CMLabels = [*range(0,nClasses,1)]
##            confMat = sklearn.metrics.confusion_matrix(grndList, predList, labels = CMLabels)
            confMat = sklearn.metrics.confusion_matrix(grndList, predList)
            
            # normalize the confusion matrix
            a = confMat.sum(axis=1).reshape((-1,1))
            confMat = confMat / a
            confMat = np.nan_to_num(confMat,nan=0.0)
            
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i,i]
            curPerClassAcc /= confMat.shape[0]
            if epoch%print_each==0:
                print('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            fn = open(log_filename,'a')
            fn.write('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}\n'.format(
                epoch_error, print2screen_avgAccRate, curPerClassAcc))
            fn.close()
            
                
            if phase=='train':
                if pgdFunc: # Projected Gradient Descent 
                    pgdFunc.PGD(model)
                      
                trackRecords['acc_train'].append(curPerClassAcc)
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                W = model.encoder.encoder.fc.weight.cpu().clone()
                tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
                trackRecords['weightNorm'].append(tmp)
                trackRecords['weights'].append(W.detach().cpu().numpy())
                
            if (phase=='val' or phase=='test') and curPerClassAcc>best_perClassAcc: #epoch_loss<best_loss:            
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name+'_best.paramOnly')
                torch.save(model.state_dict(), path_to_save_param)
                
                file_to_note_bestModel = os.path.join(work_dir, model_name+'_note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.5f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}.\n'.format(
                    epoch+1, best_loss, print2screen_avgAccRate, best_perClassAcc))
                fn.close()
                
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights    
    model = model.load_state_dict(best_model_wts)
    
    return model, trackRecords
