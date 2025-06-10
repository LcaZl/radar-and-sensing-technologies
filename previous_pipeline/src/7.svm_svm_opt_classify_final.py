# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:13:30 2021

@author: rslab
"""


import scipy.io as sio
from osgeo import gdal
import numpy as np
from sklearn import svm, model_selection
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import sys
import getopt
import json
import time
import math
import os
from sklearn.decomposition import PCA
import cv2 as cv
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import ntpath

def creategeotiff(name, array, NDV, dataSorce, dataType):
    print('i am in creategeotiff')
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], array.shape[2], dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection())
    for i in range(0, array.shape[2]):
        dataSet.GetRasterBand(i+1).WriteArray(array[:, :, i])
    dataSet.FlushCache()
    
    return name
 
    

def creategeotiffOneBand(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection()) 
    dataSet.GetRasterBand(1).WriteArray(array[:, :])
    dataSet.FlushCache()
    
    return name    


def normalize_between_0_1(tab):
    tab *= (1.0/tab.max())
    
    return tab


def readImgsPaths(training_set_path,year):
   # n_imgs = os.listdir(training_set_path)
    n_imgs = (glob.glob(training_set_path + '*' + year + '*.tif'))
    return n_imgs


def readTexturePaths(training_set_path):
   # n_imgs = os.listdir(training_set_path)
    n_imgs = (glob.glob(training_set_path + 'z*.tif'))
    return n_imgs

   
def my_sorter(elem):
    index = elem.split('_')[2]
    index = index.replace('.tif','')

    if len(index) ==1:
        index = '0' + index
    print(index)
    return (index)

def buildFullPath(img_path, n_imgs):
    new_paths = []
    for name in n_imgs:
        new_paths.append(os.path.join(img_path, name))
    return new_paths


def loadTimeSeries3D2(n_imgs, pix_region, patchSizeX, patchSizeY):
    img_temp_ = gdal.Open(n_imgs[0])
    bands =  img_temp_.RasterCount
    
    ts_3D = np.zeros((patchSizeX* patchSizeY, len(n_imgs), bands))
    for k in range(len(n_imgs)):       
            img_temp_ = gdal.Open(n_imgs[k])
            img_temp = np.asarray(img_temp_.ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY))
            img_temp = np.moveaxis(img_temp, 0, -1)
            nl = img_temp.shape[0]
            nc = img_temp.shape[1]
            nb = img_temp_.RasterCount       
            img_temp =  np.reshape(img_temp, (nc*nl, nb))
            ts_3D[:, k, : ] = img_temp
            
    img_temp = None
    img_temp_ = None
    del img_temp_
            
    return ts_3D


def labelsToNumbers(labels):
    for i in range(len(labels)):
        newLabels = labels/10
        
    return newLabels


def window_stdev(X, window_size):
    r,c = X.shape
    X+=np.random.rand(r,c)*1e-6
    diff =  np.zeros((r,c))
    while((diff <= 0).any() ):
        c1 = uniform_filter(X, window_size, mode='reflect')
        c2 = uniform_filter(X*X, window_size, mode='reflect')
        diff = c2 - c1*c1
        indexes = (diff <= 0)
        diff[indexes] = 0.0001
    return np.sqrt(diff)

def readTexture(n_imgs):
      img_temp_ds = gdal.Open(n_imgs[0])
      img_temp = np.asarray(img_temp_ds.ReadAsArray(0, 0, int(img_temp_ds.RasterXSize),int(img_temp_ds.RasterYSize)))
      img_temp = np.moveaxis(img_temp, 0, -1) # -> 2D array
      nb = img_temp_ds.RasterCount
      nr = img_temp_ds.RasterXSize
      nc = img_temp_ds.RasterYSize
     
      result_array = np.zeros((nc,nr,nb))
      
      for i in range(nb):
        band = img_temp[:,:,i]
        band[ band > np.quantile(band,0.999) ] = np.quantile( band, 0.999);
        band = (band - np.min(band))/(np.max(band)-np.min(band))
        result_array[:,:,i] = band
     
      result_array = np.reshape(result_array, (result_array.shape[0]*result_array.shape[1], result_array.shape[2]))
      indexes  = np.random.randint(len(result_array), size=(24000))
      partArray = result_array[indexes, :]    

      pca = PCA()
      pca.fit(partArray)
      coeff = np.transpose(pca.components_)
    
    
      result_array = np.matmul(result_array, coeff[:,0])
      result_array = (result_array-np.min(result_array))/np.ptp(result_array)
      result_array = np.reshape(result_array, (img_temp.shape[0], img_temp.shape[1]))


      std = window_stdev(result_array,3)
      std = std*np.sqrt(9./8)
      # std = np.moveaxis(std, 0, -1)
      kernel = np.ones((3,3),np.uint8)
      gradient = cv.morphologyEx(result_array, cv.MORPH_GRADIENT, kernel) #rangeFilt
      #gradient = np.moveaxis(gradient, 0, -1)   
      
      img_temp = None
      img_temp_ds = None
      del img_temp_ds
         
      return std, gradient



def classify(model_name, texture_flag, n_imgs, glcm, img_path):    
    print('I am in classify')
    try:
            file = open(model_name, 'rb')
            model = pickle.load(file)
            print('model: ', model)
            minValue = pickle.load(file)
            print('minValue: ', minValue)
            maxValue = pickle.load(file)
            print('maxValue: ', maxValue)
            n_classes = pickle.load(file)
            print('n_classes: ', n_classes)
            xTrain = pickle.load(file)
            print('xTrain: ', xTrain)
            labels = pickle.load(file)
            print('labels: ', labels)
            file.close()
            print("Loaded model from disk")        
    except:
            raise IOError("Unable to open model!")
            
    if texture_flag == 1:      
        std, gradient = readTexture(n_imgs)
        print('got std and gradient')  

    img_temp_ds = gdal.Open(n_imgs[0])
    xsize =  img_temp_ds.RasterXSize
    ysize =  img_temp_ds.RasterYSize
    patchSizeX = xsize // 5
    patchSizeY = ysize // 5
    patchSizeXorig = xsize // 5
    patchSizeYorig = ysize // 5   
    #mapPredictedFinal = np.zeros((ysize, xsize), dtype=int)
    probMap = np.zeros(((ysize, xsize, len(n_classes))))
    print('got probMap')
   
    for i in range(5):     #IW 5
        for j in range(5): #IW
              
            #    print('doing stuff')
            #    # #Landsat problem with irregular images
            if i==4 and patchSizeX*5 != xsize:
                patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig
            else: 
                patchSizeX = math.floor(xsize/5)

            if j==4 and patchSizeY*5 != ysize:
                patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig 
            else: 
                patchSizeY = math.floor(ysize/5)

            pix_region = np.array([ i*patchSizeXorig, j*patchSizeYorig])
            ts_2D = loadTimeSeries3D2(n_imgs, pix_region, patchSizeX, patchSizeY)
            ts_2D = np.reshape(ts_2D, (ts_2D.shape[0], ts_2D.shape[1]*ts_2D.shape[2]))

            if glcm == 1: #add GLCM features
                text_imgs = readTexturePaths(img_path)    
                texture_2D = loadTimeSeries3D2(text_imgs, pix_region, patchSizeX, patchSizeY)
                texture_2D = np.reshape(texture_2D, (texture_2D.shape[0], texture_2D.shape[1]*texture_2D.shape[2]))
                ts_2D = np.append(ts_2D, texture_2D, axis=1)
            
            mask = np.all((ts_2D == 0), axis=1)^1
            mask = np.reshape(mask, (patchSizeY, patchSizeX) )  
            
            if texture_flag == 1: #add texture features
                temp_std = np.reshape(std[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig + patchSizeX], [patchSizeX*patchSizeY, 1])
                temp2_gradient = np.reshape(gradient[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX], [patchSizeX*patchSizeY, 1])
                ts_2D = np.append(ts_2D, temp_std, axis=1)
                ts_2D = np.append(ts_2D, temp2_gradient, axis=1)

            #Normalizzazione
            ts_2D  = (ts_2D - minValue )/(maxValue-minValue)
            ts_2D[ts_2D<0]=0
            ts_2D[ts_2D>1]=1
            ts_2D = np.nan_to_num(ts_2D)
            
            #predictLabel = model.predict(ts_2D)     # IW
            print('going to predict')         
            probability = model.predict_proba(ts_2D)
            print('predicted')
            
            probability_r = np.reshape(probability, (patchSizeY, patchSizeX, probability.shape[1]))
            #predictLabel = np.reshape(predictLabel, (patchSizeY, patchSizeX))  # IW   
            
            #predictLabel = predictLabel*mask
            mask = np.expand_dims(mask, axis=2)
            probability_r= probability_r*mask
            
            #mapPredictedFinal[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX] = predictLabel  #IW
            probMap[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX, :] =probability_r       
                   
    probMap *= 254 
    probMap += 1
    probMap[np.all((probMap == 1), axis=2)]=0   
    probMap = np.uint8(np.round(probMap))
    img_temp_ds = None
    del img_temp_ds
    
    return probMap, n_classes     
 
"""def processInput(i,j,minValue,maxValue,patchSizeX,xsize,patchSizeXorig,patchSizeY,ysize,patchSizeYorig,n_imgs,texture_flag,std,gradient,model, glcm, img_path):
  #print('doing stuff')
  print('indexes: ',i,j)
  #print(globalprobMap)
  # #Landsat problem with irregular images
  if i==4 and patchSizeX*5 != xsize:
      patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig
  else: 
      patchSizeX = math.floor(xsize/5)

  if j==4 and patchSizeY*5 != ysize:
      patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig 
  else: 
      patchSizeY = math.floor(ysize/5)

  pix_region = np.array([ i*patchSizeXorig, j*patchSizeYorig])
  ts_2D = loadTimeSeries3D2(n_imgs, pix_region, patchSizeX, patchSizeY)
  ts_2D = np.reshape(ts_2D, (ts_2D.shape[0], ts_2D.shape[1]*ts_2D.shape[2]))
  
  if glcm == 1:
      text_imgs = readTexturePaths(img_path)    
      texture_2D = loadTimeSeries3D2(text_imgs, pix_region, patchSizeX, patchSizeY)
      texture_2D = np.reshape(texture_2D, (texture_2D.shape[0], texture_2D.shape[1]*texture_2D.shape[2]))
      ts_2D = np.append(ts_2D, texture_2D, axis=1)
      
  mask = np.all((ts_2D == 0), axis=1)^1
  mask = np.reshape(mask, (patchSizeY, patchSizeX) )  
  
  if texture_flag == 1:
      temp_std = np.reshape(std[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig + patchSizeX], [patchSizeX*patchSizeY, 1])
      temp2_gradient = np.reshape(gradient[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX], [patchSizeX*patchSizeY, 1])
      ts_2D = np.append(ts_2D, temp_std, axis=1)
      ts_2D = np.append(ts_2D, temp2_gradient, axis=1)


  ts_2D  = (ts_2D - minValue )/(maxValue-minValue)
  ts_2D[ts_2D<0]=0
  ts_2D[ts_2D>1]=1
  ts_2D = np.nan_to_num(ts_2D) #FR
  #predictLabel = model.predict(ts_2D)     # IW
  print('going to predict')         
  probability = model.predict_proba(ts_2D)
  print('predicted')
  
  probability_r = np.reshape(probability, (patchSizeY, patchSizeX, probability.shape[1]))
  #predictLabel = np.reshape(predictLabel, (patchSizeY, patchSizeX))  # IW   
  
  #predictLabel = predictLabel*mask
  mask = np.expand_dims(mask, axis=2)
  probability_r= probability_r*mask
  
  #mapPredictedFinal[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX] = predictLabel  #IW
  globalprobMap[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX, :] =probability_r       
"""
def processInput(i, j, minValue, maxValue, patchSizeX, xsize, patchSizeXorig, patchSizeY, ysize, patchSizeYorig, n_imgs, texture_flag, std, gradient, model, glcm, img_path):
    import gc
    import math

    if i == 4 and patchSizeX * 10 != xsize:
        patchSizeX = (xsize - patchSizeXorig * 10) + patchSizeXorig
    else:
        patchSizeX = math.floor(xsize / 10)

    if j == 4 and patchSizeY * 10 != ysize:
        patchSizeY = (ysize - patchSizeYorig * 10) + patchSizeYorig
    else:
        patchSizeY = math.floor(ysize / 10)

    pix_region = np.array([i * patchSizeXorig, j * patchSizeYorig])
    ts_2D = loadTimeSeries3D2(n_imgs, pix_region, patchSizeX, patchSizeY)
    ts_2D = np.reshape(ts_2D, (ts_2D.shape[0], -1))

    if glcm == 1:
        text_imgs = readTexturePaths(img_path)
        texture_2D = loadTimeSeries3D2(text_imgs, pix_region, patchSizeX, patchSizeY)
        texture_2D = np.reshape(texture_2D, (texture_2D.shape[0], -1))
        ts_2D = np.concatenate([ts_2D, texture_2D], axis=1)

    mask = np.any(ts_2D != 0, axis=1)
    mask = np.reshape(mask, (patchSizeY, patchSizeX))

    if texture_flag == 1:
        temp_std = std[j * patchSizeYorig:j * patchSizeYorig + patchSizeY, i * patchSizeXorig:i * patchSizeXorig + patchSizeX].reshape(-1, 1)
        temp_grad = gradient[j * patchSizeYorig:j * patchSizeYorig + patchSizeY, i * patchSizeXorig:i * patchSizeXorig + patchSizeX].reshape(-1, 1)
        ts_2D = np.concatenate([ts_2D, temp_std, temp_grad], axis=1)

    ts_2D = np.clip((ts_2D - minValue) / (maxValue - minValue), 0, 1)
    ts_2D = np.nan_to_num(ts_2D)

    probability = model.predict_proba(ts_2D)
    probability_r = np.reshape(probability, (patchSizeY, patchSizeX, probability.shape[1]))
    probability_r *= mask[:, :, np.newaxis]

    del ts_2D, probability
    gc.collect()

    return (i, j, probability_r)

def parallelclassify(model_name, texture_flag, n_imgs, glcm, img_path):    
    print('I am in parallelclassify')
    import gc
    try:
            file = open(model_name, 'rb')
            model = pickle.load(file)
            print('model: ', model)
            minValue = pickle.load(file)
            print('minValue: ', minValue)
            maxValue = pickle.load(file)
            print('maxValue: ', maxValue)
            n_classes = pickle.load(file)
            print('n_classes: ', n_classes)
            xTrain = pickle.load(file)
            print('xTrain: ', xTrain)
            labels = pickle.load(file)
            print('labels: ', labels)
            file.close()
            print("Loaded model from disk")        
    except:
            raise IOError("Unable to open model!")
            
    if texture_flag == 1:      
        std, gradient = readTexture(n_imgs)
        print('got std and gradient')
    else:
        std, gradient = None, None
        
    img_temp_ds = gdal.Open(n_imgs[0])
    xsize =  img_temp_ds.RasterXSize
    ysize =  img_temp_ds.RasterYSize
    patchSizeX = math.floor(xsize/10)
    patchSizeY = math.floor(ysize/10)
    patchSizeXorig = math.floor(xsize/10)
    patchSizeYorig = math.floor(ysize/10)   
    #mapPredictedFinal = np.zeros((ysize, xsize), dtype=int)
    globalprobMap = np.zeros((ysize, xsize, len(n_classes)), dtype=np.float32)

    for i in range(10):
        results = Parallel(n_jobs=2, verbose=2)(
            delayed(processInput)(
                i, j, minValue, maxValue, patchSizeX, xsize, patchSizeXorig,
                patchSizeY, ysize, patchSizeYorig, n_imgs, texture_flag,
                std, gradient, model, glcm, img_path
            )
            for j in range(10)
        )

        for i_, j_, prob_block in results:
            px = patchSizeXorig
            py = patchSizeYorig
            start_x = i_ * px
            end_x = start_x + prob_block.shape[1]
            start_y = j_ * py
            end_y = start_y + prob_block.shape[0]
            globalprobMap[start_y:end_y, start_x:end_x, :] = prob_block

    globalprobMap *= 254
    globalprobMap += 1
    globalprobMap[np.all(globalprobMap == 1, axis=2)] = 0
    globalprobMap = np.uint8(np.round(globalprobMap))

    del model, std, gradient, img_temp_ds
    gc.collect()

    return globalprobMap, n_classes
    """ 
    global globalprobMap
    globalprobMap = np.zeros(((ysize, xsize, len(n_classes))))
    print('created probMap')
    for i in range(5):
        print('i: ', i)
        with Parallel(n_jobs=2, verbose=2, backend='threading') as parallel:
            results = parallel(
                delayed(processInput)
                (i,j,minValue,maxValue,patchSizeX,xsize,patchSizeXorig,patchSizeY,ysize,patchSizeYorig,n_imgs,texture_flag,std,gradient,model, glcm, img_path)
                for j in range(5))
       
        #for i in range(5):     #IW 5
        #for j in range(5): #IW
        #inputs = range(5) 
        # patchSizeX xsize patchSizeXorig patchSizeY ysize patchSizeYorig n_imgs texture_flag std gradient
        #results = Parallel(n_jobs=3, require='sharedmem')(delayed(processInput)(i,j,minValue,maxValue,patchSizeX,xsize,patchSizeXorig,patchSizeY,ysize,patchSizeYorig,n_imgs,texture_flag,std,gradient,model, glcm, img_path) for j in inputs)      
                   
    globalprobMap *= 254 
    globalprobMap += 1
    globalprobMap[np.all((globalprobMap == 1), axis=2)]=0   
    globalprobMap = np.uint8(np.round(globalprobMap))
    img_temp_ds = None
    del img_temp_ds
    
    return globalprobMap, n_classes   
    """  




def saveOutputFiles(n_imgs, n_classes, output_path, probMap, tile, pathL, rowL, year, area):
    print('i am in saveOutput files')
    data = {}
 
    if int(year) == 2019:
        outputJsonPath =  os.path.join(output_path, str(tile) + "_" + str(year) + "_probabilityMapNoTexture10595.json")
        outoput_nameProb = os.path.join(output_path, str(tile) + "_" + str(year) +  "_probabilityMapNoTexture10595.tif")

        data['file_name'] = outoput_nameProb
        data['channel_num'] = int(len(n_classes))
        data['OPT_class'] = [int(i) for i in n_classes.tolist()]
        data['area'] = area
        data['year'] = int(year)
        data['type'] = "static"
        data['source'] = "Sentinel-2"
        data['tile'] = tile

    else:
        
        outputJsonPath =  os.path.join(output_path,  str(tile) + "_" + str(year) + "_probabilityMap.json")
        outoput_nameProb = os.path.join(output_path, str(tile) + "_"+ str(year) +  "_probabilityMap.tif")

        data['file_name'] = outoput_nameProb
        data['channel_num'] = int(len(n_classes))
        data['OPT_class'] = [int(i) for i in n_classes.tolist()]
        data['area'] = area
        data['year'] = int(year)
        data['type'] = "dynamic"
        data['source'] = "Landsat"
        data['tile'] = tile

    
    with open(outputJsonPath, 'w') as outfile:
        json.dump(data, outfile)
        
    img_temp_ds = gdal.Open(n_imgs[len(n_imgs)-1])
    creategeotiff(outoput_nameProb, probMap, 0, img_temp_ds, gdal.GDT_Byte)



def validateModel(model_name, validation_path, std, ts):   
    # read the model  
    try:
        file = open(model_name, 'rb')
        model = pickle.load(file)
        minValue = pickle.load(file)
        maxValue = pickle.load(file)
        n_classes = pickle.load(file)
        xTrain = pickle.load(file)
        labels = pickle.load(file)
        file.close()
        print("Loaded model from disk")        
    except:
        raise IOError("Unable to open model!")
    # read validation set 
    n_trainSets = (glob.glob(validation_path + '/*.sav'))
    print(n_trainSets)
    print(labels)
    xTest = []
    labelsTest = []
    for i in range(len(n_trainSets)):
            try:
               print(n_trainSets[i])
               file = open(n_trainSets[i], 'rb')
               xTest.append(pickle.load(file))
               labelsTest.append(pickle.load(file))            
               file.close()
               print(" Train set loaded")        
            except:
               raise IOError("Unable to load train set!")
           
    print(labelsTest)
    labelsTest = np.concatenate( labelsTest, axis=0 )
    if ts:
        for i, data in enumerate(xTest):
            xTest[i] = (data - minValue[i] )/maxValue[i]
    xTest = np.concatenate( xTest, axis=0 )
    print(xTest)
    # predict
    if not ts:
        if std:
            xTest  = (xTest - minValue )/maxValue #min and max are instead mean and std
        else:
            xTest  = (xTest - minValue )/(maxValue-minValue)
            xTest[xTest<0]=0
            xTest[xTest>1]=1
    try:
        predictLabel = model.predict(xTest)
    except:
        print('Sample with 0 test')
    accuracy = accuracy_score(labelsTest, predictLabel)            
    f1 = f1_score(labelsTest, predictLabel, average=None)
    priorTrain = np.unique(labels, return_counts=True)
    priorValidation = np.unique(labelsTest, return_counts=True)
    print('Overall train-set prior: \n', priorTrain[0], '\n', priorTrain[1])  
    print('Overall val-set prior: \n', priorValidation[0], '\n', priorValidation[1])       
    print('Overall Accuracy: ', accuracy)
    print('Overall F1_score: ', f1)
    # validation tile bases 
    for i in range(len(n_trainSets)):
        try:
               file = open(n_trainSets[i], 'rb')
               xTest2 = np.array(pickle.load(file))
               labelsTest2 = pickle.load(file)         
               file.close()
               print(" Train set loaded")        
        except:
               raise IOError("Unable to load train set!")
        if not ts:
            if std:
                xTest2  = (xTest2 - minValue )/maxValue #min and max are instead mean and std
            else:
                xTest2  = (xTest2 - minValue )/(maxValue-minValue)
                xTest2[xTest2<0]=0
                xTest2[xTest2>1]=1
        else:
            xTest2 = (xTest2 - minValue[i] )/maxValue[i]
        try:
            predictLabel = model.predict(xTest2)
        except:
            print('Provlem with ', n_trainSets[i])
        accuracy = accuracy_score(labelsTest2, predictLabel)            
        f1 = f1_score(labelsTest2, predictLabel, average=None)
        print(ntpath.basename(n_trainSets[i]), 'Accuracy: ', accuracy)
        print(ntpath.basename(n_trainSets[i]), 'F1_score: ', f1)

    
def Help():
    print("\nThis script performs  SVM  classification of the time series of the multispectral data and returns posterior probability."
          "Required parameters :\n"\
          "  -i     the path to the time series of images to be processed \n"\
          "  -m     the path to the SVM model \n"
          "  -o     the path where the posterior probability matrix will be saved \n"
          "  -t     the tile to be processed (specify only for Sentinel) \n"\
          "  -p     the path of the Landsat image to be processed (specify only for Landsat) \n"\
          "  -r     the row of the Landsat image to be processed (specify only for Landsat) \n"\
          "  -y     year to be classified \n"\
          "  -a     area to be classified (africa, amazonia, siberia) \n"
          "  -f     texture flag if set to 1 process the textural features (std and gradient)  \n"
          "  -g     glcm flag if set to 1 process the glcm features  \n"
          "  -v     the path to the folder where the validsation .sav files are saved, if the script is run to classify leave it empty\n"          

          )
    sys.exit(2)



def classify_main(img_path, model_name, output_path, year, area, tile=0, pathL=0, rowL=0, texture_flag=0, glcm=0, validation_path=0, multicore=1):

        startTime = datetime.now()
        
        n_imgs = readImgsPaths(img_path, year)
        print('n_imgs: ', n_imgs)

        n_imgs.sort(key=my_sorter)
        n_imgs = buildFullPath(img_path,n_imgs)     
        print('n_imgs: ', n_imgs)
        
        if(validation_path): # run the code to validate the model
          validateModel(model_name, validation_path)
          print('execution_time:',datetime.now() - startTime)

        elif multicore:
          probMap, n_classes = parallelclassify(model_name, texture_flag, n_imgs, glcm, img_path)
          saveOutputFiles(n_imgs, n_classes, output_path, probMap, tile, pathL, rowL, year, area)    
          print('execution_time:',datetime.now() - startTime)
          
        else:
          probMap, n_classes = classify(model_name, texture_flag, n_imgs, glcm, img_path)     
          saveOutputFiles(n_imgs, n_classes, output_path, probMap, tile, pathL, rowL, year, area)    
          print('execution_time:',datetime.now() - startTime)

from utility import *
from threading import Thread

if __name__ == "__main__":
    
    configuration = {}
    configuration['output_folder_path'] = "/home/hrlcuser/media/SVM_old/classification/"
    os.makedirs(configuration['output_folder_path'], exist_ok=True)
    # Parameters setup
    performance_file = setup_logger(configuration)
    
    # Start memory monitoring in a separate thread
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    tile = 0
    pathL = 0
    rowL = 0
    texture_flag = 0
    glcm = 1
    validation_path =  ""
        
##    try:
##        opts, args = getopt.getopt(sys.argv[1:],"i:m:o:t:p:r:y:a:c:f:g:v",["help"])
##    except getopt.GetoptError:
##        print('Invalid argument!')
##        sys.exit(2)
##    for opt, arg in opts:
##        if opt in ("--help"):
##            Help()
##        elif opt in ("-i"):
##            img_path = arg
##        elif opt in ("-m"):
##            model_name = arg
##        elif opt in ("-o"):
##            output_path = arg
##        elif opt in ("-t"):
##            tile = arg
##        elif opt in ("-p"):
##            pathL = arg
##        elif opt in ("-r"):
##            rowL = arg
##        elif opt in ("-y"):
##            year = arg
##        elif opt in ("-a"):
##            area = arg
##        elif opt in ("-f"):
##            texture_flag = arg
##        elif opt in ("-g"):
##            glcm = arg
##        elif opt in ("-v"):
##            validation_path = arg
##        elif opt in ("-c"):
##            multicore = arg
##        if (not opts):
##            Help()

    img_path = '/home/hrlcuser/media/S2_processed_composites/median_composites_restored_adj/'
    model_names = [
##        '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KUQ_round.sav',
##        '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KUQ_round_std.sav',
##        '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KXT_round.sav',
##        '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KXT_round_std.sav',
##        '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KXT_round_std_time-reduced.sav',
        '/home/hrlcuser/media/SVM_old/trained_SVM.sav'
    ]

    validation_paths = [
##        '7.svm-opt-classify-final_test-data/validation/20KNA',
##        '7.svm-opt-classify-final_test-data/validation/20KPF',
##        '7.svm-opt-classify-final_test-data/validation/21KUQ',
##        '7.svm-opt-classify-final_test-data/validation/21KXT',
##        '7.svm-opt-classify-final_test-data/validation_round/20KNA',
##        '7.svm-opt-classify-final_test-data/validation_round/20KPF',
##        '7.svm-opt-classify-final_test-data/validation_round/21KUQ',
##        '7.svm-opt-classify-final_test-data/validation_round/21KUQ',
##        '7.svm-opt-classify-final_test-data/validation_round/21KUQ',
##        '7.svm-opt-classify-final_test-data/validation_round/21KUQ',
##        '7.svm-opt-classify-final_test-data/validation_round/21KUQ',
        #'7.svm-opt-classify-final_test-data/val_sav',
        '/home/hrlcuser/media/S2_processed_composites/median_composites_restored_adj'
    ]
    standardization = [False] #,True]
    tile_specific = [False] #,True]
    year = '2019'
    area = 'Amazzonia'  
    n_imgs = readImgsPaths(img_path, year)   
    print(n_imgs)
    n_imgs.sort(key=my_sorter)
    n_imgs = buildFullPath(img_path,n_imgs)   
    output_path = '/home/hrlcuser/media/SVM_old'
    print('n_imgs: ', n_imgs)

    np.set_printoptions(precision=4, suppress=True)
    for model_name, validation_path, std, ts in zip(model_names,validation_paths,standardization,tile_specific):
        name = os.path.basename(model_name)[:-4]
##        for i, validation_path in enumerate(validation_paths):
##        val_name = os.path.basename(validation_path)
        print(f'\n\n\t  MODEL: {name}')
        startTime = datetime.datetime.now()
        
        if(validation_path): # run the code to validate the model
            validateModel(model_name, validation_path, std, ts)
            print('execution_time:',datetime.datetime.now() - startTime)


        probMap, n_classes = parallelclassify(model_name, texture_flag, n_imgs, glcm, img_path)
        saveOutputFiles(n_imgs, n_classes, output_path, probMap, tile, pathL, rowL, year, area)
        print('execution_time:',datetime.datetime.now() - startTime)
        # -> ValueError: operands could not be broadcast together with shapes (4822416,144) (152,) 
        
        
        #probMap, n_classes = classify(model_name, texture_flag, n_imgs, glcm, img_path)
        #saveOutputFiles(n_imgs, n_classes, output_path, probMap, tile, pathL, rowL, year, area)
        #print('execution_time:',datetime.now() - startTime)
        # -> numpy.core._exceptions._ArrayMemoryError: Unable to allocate 129. GiB for an array with shape (120560400, 12, 12) and data type float64