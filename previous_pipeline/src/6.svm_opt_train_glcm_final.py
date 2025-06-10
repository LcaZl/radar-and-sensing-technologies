# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:44:11 2021

@author: rslab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:33:15 2021

@author: rslab
"""

import getopt
import glob
import os
import pickle
import random
import sys

import cv2 as cv
import dask.array as da
##import gdal
from osgeo import gdal, ogr, osr
import matplotlib.path as mplPath
import numpy as np
##import ogr
##import osr
import rasterio
from joblib import Parallel, cpu_count, delayed
from numpy import cov, dot, linalg, mean
from numpy.lib.stride_tricks import as_strided as ast
from osgeo import ogr
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
from sklearn import model_selection, svm
from sklearn.decomposition import PCA
import ntpath

from datetime import datetime

def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print('GDAL Message:', file=sys.stderr)
    print('\tNumber: %s' % (err_num), file=sys.stderr)
    print('\tType: %s' % (err_class), file=sys.stderr)
    print('\tDescription: %s' % (err_msg), file=sys.stderr)


def readImgsPaths(training_set_path):
   # n_imgs = os.listdir(training_set_path)
    n_imgs = (glob.glob(training_set_path + '*.tif'))
    return [os.path.basename(img_path) for img_path in n_imgs]
   
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
     

def labelsToNumbers(labels):
    for i in range(len(labels)):
        newLabels = labels/10
        
    return newLabels



def defineTrainSVM_std( xTrain, labels, ts):

    #Normalizzazione TS?
    if ts:
        meanValues = None
        stdValues = None
    else:
        meanValues = np.mean(xTrain,0)
        stdValues = np.std(xTrain,0)
        xTrain = (xTrain - meanValues )/stdValues
##    xTrain[xTrain<0]=0
##    xTrain[xTrain>1]=1
    
    gSpace = np.linspace(1e-3,10,10)
    cSpace = np.linspace(100,1000,5)
    bestcv = 1
    for gIndex in range(len(gSpace)):
        g = gSpace[gIndex]
        for cIndex in range(len(cSpace)):
            c = cSpace[cIndex]
            clf = svm.SVC(C=c, gamma=g, probability=True)
            scores = model_selection.cross_val_score(clf,xTrain,labels,cv=3)
            cv = scores.mean() * 100
            if cv >= bestcv:
                 bestcv = cv
                 bestc = c
                 bestg = g       
                 print(cv)
    clf = svm.SVC(C=bestc,gamma=bestg, probability=True).fit(xTrain, labels)

    return clf, meanValues, stdValues

def defineTrainSVMparallel_std( xTrain, labels):

    n = os.cpu_count()//2
    meanValues = np.mean(xTrain,0)
    stdValues = np.std(xTrain,0)
    xTrain = (xTrain - meanValues )/stdValues
    
    gSpace = np.linspace(1e-3,10,10)
    cSpace = np.linspace(100,1000,5)
    bestcv = 1
    clf_list = []

    for gIndex in range(len(gSpace)):
        g = gSpace[gIndex]
        for cIndex in range(len(cSpace)):
            c = cSpace[cIndex]
            clf = svm.SVC(C=c, gamma=g, probability=True)
            clf_list.append((clf,g,c))

    results = Parallel(n_jobs=n)(delayed(get_score)(clf_values, xTrain, labels) for clf_values in clf_list)

    for res in results:
        cv = res['cv']
        if cv >= bestcv:
            bestcv = cv
            bestc = res['c']
            bestg = res['g']  
            print(cv)

    clf = svm.SVC(C=bestc,gamma=bestg, probability=True).fit(xTrain, labels)
    
    return clf, meanValues, stdValues

def defineTrainSVM( xTrain, labels):

    minValues = 0
    maxValues = np.quantile(xTrain,0.999,0)
    #Normalizzazione TS?
    
    xTrain = (xTrain - minValues )/(maxValues-minValues)
    xTrain[xTrain<0]=0
    xTrain[xTrain>1]=1
    
    gSpace = np.linspace(1e-3,10,10)
    cSpace = np.linspace(100,1000,5)
    bestcv = 1
    for gIndex in range(len(gSpace)):
        g = gSpace[gIndex]
        for cIndex in range(len(cSpace)):
            c = cSpace[cIndex]
            clf = svm.SVC(C=c, gamma=g, probability=True)
            scores = model_selection.cross_val_score(clf,xTrain,labels,cv=3)
            cv = scores.mean() * 100
            if cv >= bestcv:
                 bestcv = cv
                 bestc = c
                 bestg = g       
                 print(cv)
    clf = svm.SVC(C=bestc,gamma=bestg, probability=True).fit(xTrain, labels)

    return clf, minValues, maxValues
 

def get_score(clf_values:tuple, xTrain, labels):

    clf = clf_values[0]
    g = clf_values[1]
    c = clf_values[2]
    scores = model_selection.cross_val_score(clf,xTrain,labels,cv=3)
    cv = scores.mean() * 100

    svm_dict = {
        'clf':clf,
        'g':g,
        'c':c,
        'cv': cv
    }

    print ('performed score with values:', g,c)
    return svm_dict


def defineTrainSVMparallel( xTrain, labels):

    n = os.cpu_count()//2
    minValues = 0
    maxValues = np.quantile(xTrain,0.999,0)
    #Normalizzazione TS?
    
    xTrain = (xTrain - minValues )/(maxValues-minValues)
    xTrain[xTrain<0]=0
    xTrain[xTrain>1]=1
    
    gSpace = np.linspace(1e-3, 10, 10)
    cSpace = np.linspace(100,1000,5)
    
    bestcv = 1
    clf_list = []

    for gIndex in range(len(gSpace)):
        g = gSpace[gIndex]
        for cIndex in range(len(cSpace)):
            c = cSpace[cIndex]
            clf = svm.SVC(C=c, gamma=g, probability=True)
            clf_list.append((clf,g,c))

    results = Parallel(n_jobs=n)(delayed(get_score)(clf_values, xTrain, labels) for clf_values in clf_list)

    for res in results:
        cv = res['cv']
        if cv >= bestcv:
            bestcv = cv
            bestc = res['c']
            bestg = res['g']  
            print(cv)

    clf = svm.SVC(C=bestc,gamma=bestg, probability=True).fit(xTrain, labels)
    
    return clf, minValues, maxValues
    
def coord2pixel(geo_transform, lat, long):
    xOffset = int((lat - geo_transform[0]) / geo_transform[1])
    yOffset = int((long - geo_transform[3]) / geo_transform[5])


    return xOffset, yOffset

def coord2pixel_round(geo_transform, lat, long):
    xOffset = round((lat - geo_transform[2]) / geo_transform[0])
    yOffset = round((long - geo_transform[5]) / geo_transform[4])


    return xOffset, yOffset

def readTrainingSetFromShp(shapefile, tif):
    file = ogr.Open(shapefile)
    layer = file.GetLayer(0)  
    ldefn = layer.GetLayerDefn()
    featureCount = layer.GetFeatureCount()
    points = np.zeros((featureCount,2+ldefn.GetFieldCount()));  
     
    img = gdal.Open(tif)
    raster_geo_transform = img.GetGeoTransform()

    
    for index in range(featureCount):
            feature = layer.GetFeature(index)
            geometry = feature.GetGeometryRef()
            
            #gt_lat, gt_long = geometry.GetX(), geometry.GetY()
            gt_long, gt_lat = geometry.GetX(), geometry.GetY()
            x, y = coord2pixel(raster_geo_transform, gt_lat, gt_long)
            
            points[index, 0] = y
            points[index, 1] = x
            points[index, 2] = feature.GetField(0)  #class_level1
            #points[index, 3] = feature.GetField(1)
            # points[index, 4] = feature.GetField(2)
            # points[index, 5] = feature.GetField(3)
            
    return points





def readTrainingSetFromShp_round(shapefile, tif):

    file = ogr.Open(shapefile)
    layer = file.GetLayer(0)  
    ldefn = layer.GetLayerDefn()
    featureCount = layer.GetFeatureCount()
    points = np.zeros((featureCount,2+ldefn.GetFieldCount()))
     
    raster = gdal.Open(tif)
    transform2 = raster.GetGeoTransform()
     
        
    with rasterio.open(tif) as src:   
         rows, cols = src.shape
         epsg_dest = src.crs
         if epsg_dest.is_epsg_code:
            epsg_dest_code = int(epsg_dest['init'].lstrip('epsg:'))
         transform = src.transform

    pixelWidth = transform[0]
    pixelHeight = transform[4]
    xLeft = transform[2]
    yTop = transform[5]
    xRight = xLeft+cols*pixelWidth
    yBottom = yTop+rows*pixelHeight  

    
    bbPath = mplPath.Path([(xLeft, yBottom),
                         (xRight, yBottom),
                         (xRight, yTop),
                         (xLeft, yTop)])
    
    
    for index in range(featureCount):
            feature = layer.GetFeature(index)
            geometry = feature.GetGeometryRef()
            if geometry != None:
##                gt_long, gt_lat = geometry.GetX(), geometry.GetY() #UNDERSTAND STRANGE BEHAVIOUR BETWEEN ME AND IWONA
                gt_lat, gt_long = geometry.GetX(), geometry.GetY()          
                x, y = reprojectShp2(gt_lat, gt_long, layer, epsg_dest_code)
                if( bbPath.contains_point((x, y ))):
                    row, col = coord2pixel_round(transform, x, y )
    
                    points[index, 0] = col   ##change it back!!! to col
                    points[index, 1] = row ##change it back!!! to row
                    points[index, 2] = feature.GetField(1)  #class_level1
                    # points[index, 3] = feature.GetField(1)  
                    # points[index, 4] = feature.GetField(2)
                    # points[index, 5] = feature.GetField(3)
    return points




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


def princomp(A):
     """ performs principal components analysis 
         (PCA) on the n-by-p data matrix A
         Rows of A correspond to observations, columns to variables. 
    
     Returns :  
      coeff :
        is a p-by-p matrix, each column containing coefficients 
        for one principal component.
      score : 
        the principal component scores; that is, the representation 
        of A in the principal component space. Rows of SCORE 
        correspond to observations, columns to components.
      latent : 
        a vector containing the eigenvalues 
        of the covariance matrix of A.
     """
     # computing eigenvalues and eigenvectors of covariance matrix
     M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
     [latent,coeff] = linalg.eig(cov(M)) # attention:not always sorted
     score = dot(coeff.T,M) # projection of the data in the new space
     
     return coeff,score,latent
 
    

def readTexture(n_imgs, xTrain, trainingSetPoints ):
      img_temp_ds = gdal.Open(n_imgs[0])
      img_temp = np.asarray(img_temp_ds.ReadAsArray(0, 0, int(img_temp_ds.RasterXSize),int(img_temp_ds.RasterYSize)))
      img_temp = np.moveaxis(img_temp, 0, -1)
      nb = img_temp_ds.RasterCount
      nr = img_temp_ds.RasterXSize
      nc = img_temp_ds.RasterYSize
     
      result_array = np.zeros((nc,nr,nb))
      for i in range(nb):
              band = img_temp[:,:,i]
              band[ band > np.quantile(band,0.999) ] = np.quantile( band, 0.999);
              band = (band - np.min(band))/(np.max(band)-np.min(band))
              result_array[:,:,i] =band
     
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
      #std = np.moveaxis(std, 0, -1)
      kernel = np.ones((3,3),np.uint8)
      gradient = cv.morphologyEx(result_array, cv.MORPH_GRADIENT, kernel) #rangeFilt
      #gradient = np.moveaxis(gradient, 0, -1)   

      #adds to the training set
      temp_std = np.zeros((len(trainingSetPoints),1))
      temp2_range = np.zeros((len(trainingSetPoints),1))

      for p in range(len(trainingSetPoints)):
             x = int(trainingSetPoints[p, 0])
             y = int(trainingSetPoints[p, 1])
             temp_std[p,0] = std[x,y]
             temp2_range[p,0] = gradient[x,y]
     
      xTrain2 = np.append(xTrain,temp_std,axis=1) 
      xTrain2 = np.append(xTrain2,temp2_range,axis=1)
      
      img_temp = None
      img_temp_ds = None
    
      return xTrain2

# GLCM
def im_resize(im,Nx,Ny):
    '''
    resize array by bivariate spline interpolation
    '''
    ny, nx = np.shape(im)
    xx = np.linspace(0,nx,Nx)
    yy = np.linspace(0,ny,Ny)

    try:
        im = da.from_array(im, chunks=1000)   #dask implementation
    except:
        pass

    newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im)
    return newKernel(yy,xx)


def p_me(z, win):
    '''
    loop to calculate graycoprops
    '''
    try:
       # glcm = graycomatrix(z2, [5], [0], 256, symmetric=True, normed=True)
        glcm = graycomatrix(z, [1], [0], symmetric=True, normed=True)
       # return (glcm)

        #uncomment
        cont = graycoprops(glcm, 'contrast')
        diss = graycoprops(glcm, 'dissimilarity')
        homo = graycoprops(glcm, 'homogeneity')
        eng = graycoprops(glcm, 'energy')
        corr = graycoprops(glcm, 'correlation')
        ASM = graycoprops(glcm, 'ASM')
        return (cont, diss, homo, eng, corr, ASM)
    except:
        return (0,0,0,0,0,0)


def read_raster(in_raster):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    #get 

    #for both Landsat and Sentinel
    data = ds.GetRasterBand(4).ReadAsArray() #7

    
    data = data.astype(np.float32)
    data[data<=0] = np.nan
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]

    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    del ds
    # create a grid of xy coordinates in the original projection
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    return data, xx, yy, gt


def creategeotiffOneBand(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], 1, dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection()) 
    dataSet.GetRasterBand(1).SetNoDataValue(np.nan)
    dataSet.GetRasterBand(1).WriteArray(array[:, :])
    dataSet.FlushCache()
    
    dataSet = None
    del dataSet
    
    return name       



def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')



def sliding_window(a, ws, ss = None, flatten = True):
    '''
    Source: http://www.johnvinyard.com/blog/?p=268#more-268
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''      
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)


    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)


    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    
   # dim = filter(lambda i : i != 1,dim) iwona commented!

    return a.reshape(dim), newshape  
    


def computeGLCM(in_raster,  outputPath):   
    
    win_sizes = [7]
    for win_size in win_sizes[:]:  

        win = win_size
        meter = str(win/4)
        merge, xx, yy, gt = read_raster(in_raster)
        merge[np.isnan(merge)] = 0   
        
      #  merge2 = merge/10000 #IWONA ADDED!
        merge = (merge - np.min(merge) )/(np.max(merge)-np.min(merge))
        merge *= 254 #iw
        merge += 1      
        z,ind = sliding_window(merge,(win,win),(win,win))
        Ny, Nx = np.shape(merge)
        

        z = z.astype(np.uint8) #iw
        w = Parallel(n_jobs = cpu_count(), verbose=0)(delayed(p_me)(z[k], win) for k in range(len(z)))
        p_me(z[0], win)
    
        cont = [a[0] for a in w]
        diss = [a[1] for a in w]
        homo = [a[2] for a in w]
        eng  = [a[3] for a in w]
        corr = [a[4] for a in w]
        ASM  = [a[5] for a in w]

        #Reshape to match number of windows
        plt_cont = np.reshape(cont , ( ind[0], ind[1] ) )
        plt_diss = np.reshape(diss , ( ind[0], ind[1] ) )
        plt_homo = np.reshape(homo , ( ind[0], ind[1] ) )
        plt_eng = np.reshape(eng , ( ind[0], ind[1] ) )
        plt_corr = np.reshape(corr , ( ind[0], ind[1] ) )
        plt_ASM =  np.reshape(ASM , ( ind[0], ind[1] ) )
        del cont, diss, homo, eng, corr, ASM

        #Resize Images to receive texture and define filenames
        contrast = im_resize(plt_cont,Nx,Ny)
        contrast[merge==0]=np.nan
        dissimilarity = im_resize(plt_diss,Nx,Ny)
        dissimilarity[merge==0]=np.nan    
        homogeneity = im_resize(plt_homo,Nx,Ny)
        homogeneity[merge==0]=np.nan
        energy = im_resize(plt_eng,Nx,Ny)
        energy[merge==0]=np.nan
        correlation = im_resize(plt_corr,Nx,Ny)
        correlation[merge==0]=np.nan
        ASM = im_resize(plt_ASM,Nx,Ny)
        ASM[merge==0]=np.nan
        del plt_cont, plt_diss, plt_homo, plt_eng, plt_corr, plt_ASM

        
        img_temp_ds = gdal.Open(in_raster)
        creategeotiffOneBand(os.path.join(outputPath, 'zcontrastParB8_13_13.tif'), contrast,  0, img_temp_ds,  gdal.GDT_Float32)
        creategeotiffOneBand(os.path.join(outputPath, 'zdissimilarityB8_13_13.tif'), dissimilarity,  0, img_temp_ds,  gdal.GDT_Float32)
        creategeotiffOneBand(os.path.join(outputPath, 'zhomogeneityB8_13_13.tif'), homogeneity,  0, img_temp_ds,  gdal.GDT_Float32)
        creategeotiffOneBand(os.path.join(outputPath, 'zenergyB8_13_13.tif'), energy,  0, img_temp_ds,  gdal.GDT_Float32)
        creategeotiffOneBand(os.path.join(outputPath, 'zcorrelationB8_13_13.tif'), correlation,  0, img_temp_ds,  gdal.GDT_Float32)
        creategeotiffOneBand(os.path.join(outputPath, 'zASMPB8_13_13.tif'), ASM,  0, img_temp_ds,  gdal.GDT_Float32)

        del contrast, merge, xx, yy, gt, meter, dissimilarity, homogeneity, energy, correlation, ASM
 
    
def splitSamplesIntoTrainValidation(xTrain, labels, validate_name):
    
        trSet = np.column_stack(( labels, xTrain))
        n_classes = np.unique(labels)
        train = []
        validate = []
        for i in range( len(n_classes)):
            temp = trSet[trSet[:,0] == n_classes[i]]
            half = int(len(temp)/2)

             #randomly select half samples 
            randomlist = random.sample(range(0, len(temp)), half)
            train.append(temp[randomlist,:])
            temp = np.delete(temp, randomlist, axis=0)
            validate.append(temp)

        train = np.concatenate( train, axis=0 )
        validate = np.concatenate( validate, axis=0 )
                    
         # save the validation set
        file = open(validate_name, 'wb')
        pickle.dump(validate[:,1:], file)
        pickle.dump(validate[:,0], file)  
        file.close()

        return train


def reprojectShp2(gt_lat, gt_long, shp_layer, epsg_dest_code):    
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(gt_lat, gt_long)

    # create coordinate transformation
    inSpatialRef = shp_layer.GetSpatialRef()

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_dest_code)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    point.Transform(coordTransform)
    
    x = point.GetX()
    y= point.GetY()
    return x, y

def coord2pixel2(geo_transform, lat, long):
    xOffset = int((lat - geo_transform[2]) / geo_transform[0])
    yOffset = int((long - geo_transform[5]) / geo_transform[4])


    return xOffset, yOffset


def readTrainingSetFromShp2(shapefile, tif):

    file = ogr.Open(shapefile)
    layer = file.GetLayer(0)  
    ldefn = layer.GetLayerDefn()
    featureCount = layer.GetFeatureCount()
    points = np.zeros((featureCount,2+ldefn.GetFieldCount()))

    raster = gdal.Open(tif)
    transform2 = raster.GetGeoTransform()
     
        
    with rasterio.open(tif) as src:   
         rows, cols = src.shape
         epsg_dest = src.crs
         if epsg_dest.is_epsg_code:
            epsg_dest_code = int(epsg_dest['init'].lstrip('epsg:'))
         transform = src.transform

    pixelWidth = transform[0]
    pixelHeight = transform[4]
    xLeft = transform[2]
    yTop = transform[5]
    xRight = xLeft+cols*pixelWidth
    yBottom = yTop+rows*pixelHeight  

    
    bbPath = mplPath.Path([(xLeft, yBottom),
                         (xRight, yBottom),
                         (xRight, yTop),
                         (xLeft, yTop)])
    
    
    for index in range(featureCount):
            feature = layer.GetFeature(index)
            geometry = feature.GetGeometryRef()
            if geometry != None:
##                gt_long, gt_lat = geometry.GetX(), geometry.GetY() #UNDERSTAND STRANGE BEHAVIOUR BETWEEN ME AND IWONA
                gt_lat, gt_long = geometry.GetX(), geometry.GetY()          
                x, y = reprojectShp2(gt_lat, gt_long, layer, epsg_dest_code)
                if( bbPath.contains_point((x, y ))):
                    row, col = coord2pixel2(transform, x, y )
    
                    points[index, 0] = col   ##change it back!!! to col
                    points[index, 1] = row ##change it back!!! to row
##                    print(feature.GetField())
                    points[index, 2] = feature.GetField(1)  #class_level1
                    # points[index, 3] = feature.GetField(1)  
                    # points[index, 4] = feature.GetField(2)
                    # points[index, 5] = feature.GetField(3)
    return points

   
def readTSforTrainingSet(n_imgs, trainingSetPoints ):

     img_temp_ds = gdal.Open(n_imgs[0])
     bands =  img_temp_ds.RasterCount
     count = 0
     if any('zcorrelationB8' in s for s in n_imgs): # when the GLCM features are included
         for c in range(len(n_imgs)):
             name = ntpath.basename(n_imgs[c])
             if name[0] == 'z':
                 count = count+1
         nc = bands*(len(n_imgs)-count)+count
     else:
         nc = bands*len(n_imgs)
         
     xTrain_list = np.zeros((len(trainingSetPoints), nc))

     labels =  np.zeros(len(trainingSetPoints))
     ix =0 
     for i in range(len(n_imgs)):
         img_temp_ds = gdal.Open(n_imgs[i])
         img_temp = np.asarray(img_temp_ds.ReadAsArray(0, 0, int(img_temp_ds.RasterXSize),int(img_temp_ds.RasterYSize)))
         img_temp = np.moveaxis(img_temp, 0, -1)
         bands =  img_temp_ds.RasterCount
          
         for p in range(len(trainingSetPoints)):
                 x = int(trainingSetPoints[p, 0])
                 y = int(trainingSetPoints[p, 1])
                 
                 xTrain_list[p,ix:ix+bands] = img_temp[x,y]
                 
         ix = ix+bands
         
     labels = trainingSetPoints[:,2]
   #  labels = labelsToNumbers(labels)  
     img_temp = None
     img_temp_ds = None
     
     return xTrain_list, labels    

def Help():  
      print("The script works in two modes: \n "
      "1)	Reads the train set. Reads the information provided by the time series of Sentinel-2 or Landsat 5/ Landsat 7 or Landsat 8 images,"\
         " according to points defined in the shapefile. Moreover, for every point in the training set computes gradient and standard deviation."\
         " Saves the output training set together with labels to .sav file.\n"
      "2)	Reads the .sav files with defined training set and trains the SVM RBF model. The model is saved to .sav file. \n \n"

      "Required parameters :\n"\
      "  -i     the path to the input of the script: for a traing set reading or glcm computation it is a path to the time series of images to be processed "
               " for a model training it is path where all the training set files(.sav) can be find\n"          
      "  -o     the path where the script output will be saved (please, indicate the path together with an output filename) \n"
      "  -s     the path to the training set (.shp), if the script is run to  train the SVM model leave it empty\n"          
      "  -g     the flag to set as 1 if to be run for GLCM  computation only\n"   
      "  -f     texture flag if set to 1 process the textural features (std and gradient)\n"     
      "  -v     the path to the folder where the validation .sav files are saved (please, indicate the path together with an output filename). Specify this path only if the shapefile comes from PoliMi. \n"          
          )
      sys.exit(2)



if __name__ == "__main__":
##        try:
##          opts, args = getopt.getopt(sys.argv[1:],"i:o:s:g:f:v:",["help"])
##        except getopt.GetoptError:
##          print('Invalid argument!')
##          sys.exit(2)
##        for opt, arg in opts:
##          if opt in ("--help"):
##              Help()
##          elif opt in ("-i"):
##              img_path = arg
##          elif opt in ("-o"):
##              output_name = arg
##          elif opt in ("-s"):
##              shapefile = arg
##          elif opt in ("-g"):
##              GLCM = arg
##          elif opt in ("-f"):
##              texture_flag = arg
##          elif opt in ("-v"):
##              validate_name = arg
##        if (not opts):
##          Help()
        
        gdal.PushErrorHandler(gdal_error_handler)
        shapefile = ""
        GLCM = None #"/home/hrlcuser/media/SVM_old/glcm_features"
        texture_flag = 0
        validate_name = ''
##        img_paths = [r'D:\Users\gianm\Documents\Ricerca post-laurea\CCI+\amazonia\unlabeled_data\20KNA/']*2+[r'D:\Users\gianm\Documents\Ricerca post-laurea\CCI+\amazonia\unlabeled_data\20KPF/']*2#+['6.svm-opt-train-glcm-final_test-data/']*2
        img_paths = ['/home/hrlcuser/media/S2_processed_composites/median_composites_restored_adj/']
        output_names = [
##            '6.svm-opt-train-glcm-final_test-data/20KNA.sav',
##            '6.svm-opt-train-glcm-final_test-data/20KNA_round.sav',
##            '6.svm-opt-train-glcm-final_test-data/20KPF.sav',
##            '6.svm-opt-train-glcm-final_test-data/20KPF_round.sav',
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_20KNA.sav',
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_20KNA_round.sav'
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_20KPF.sav',
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_20KPF_round.sav'
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KUQ.sav',
##            '6.svm-opt-train-glcm-final_test-data/trained_SVM_21KUQ_round.sav',
            '/home/hrlcuser/media/SVM_old/trained_SVM.sav',
            #'6.svm-opt-train-glcm-final_test-data/trained_SVM_std_tile-specific.sav',
            ]
        shapefiles = [''] #['/home/hrlcuser/media/data/training_points/amazonia_static_area_2019_photointerpreted_UniTN.shp']
##        shapefiles = ['6.svm-opt-train-glcm-final_test-data/CCI_HRLC_Training_Points/static_area_2019/Amazon_static_area_2019/Amazon_static_area_2019.shp']*4#+['','']
        roundings = [None] #,None]
        standardization = [False] #, True]
        tile_specific = [False] #,True]
        for img_path, output_name, shapefile, rounding, std, ts in zip(img_paths,output_names,shapefiles,
                                                                       roundings,standardization,tile_specific):
           ######################################################################
           # compute GLCM features
           ######################################################################
           
            if GLCM: 
                print("primo ramo")
                
                n_imgs = readImgsPaths(img_path)    
                n_imgs.sort(key=my_sorter)
                n_imgs = buildFullPath(img_path,n_imgs)
                #n_imgs = sorted(n_imgs)

                computeGLCM(n_imgs[0], img_path) 
           
           ######################################################################
           # 1) read train set points and save it
           ######################################################################
            elif shapefile:  
                print("secondo ramo")
                print(shapefile,img_path)
                n_imgs = readImgsPaths(img_path)    
                n_imgs.sort(key=my_sorter)
                n_imgs = buildFullPath(img_path,n_imgs)
                #n_imgs = sorted(n_imgs)

                if rounding:
                    trainingSetPoints = readTrainingSetFromShp_round(shapefile, n_imgs[0])
                else:
                    trainingSetPoints = readTrainingSetFromShp2(shapefile, n_imgs[0])
                trainingSetPoints = trainingSetPoints[~np.all(trainingSetPoints == 0, axis=1)] #removes all the elements equal to 0
                if len(trainingSetPoints) !=0:
                    xTrain, labels = readTSforTrainingSet(n_imgs, trainingSetPoints)
                    labels = labels[~np.all(xTrain == 0, axis=1)] 
                    trainingSetPoints = trainingSetPoints[~np.all(xTrain == 0, axis=1)]
                    xTrain = xTrain[~np.all(xTrain == 0, axis=1)] #removes all the elements equal to 0

                    if texture_flag == 1:
                        xTrain = readTexture(n_imgs, xTrain, trainingSetPoints)
                        
                    #######################################################################################################  
                    # split PoliMi (Gorica) training set into two: half samples for training and other half for validation   
                    #######################################################################################################
                    print(f"Labels:", np.unique(labels, return_counts = True), "\n", labels)
                    
                    if validate_name: 
                        train = splitSamplesIntoTrainValidation(xTrain, labels)                   
                        labels = train[:,0]
                        xTrain = train[:,1:]

                    # save the training set
                    file = open(os.path.join(img_path, 'dataset.sav'), 'wb')
                    pickle.dump(xTrain, file)
                    print(f"Labels:", np.unique(labels, return_counts = True), "\n", labels)
                    pickle.dump(labels, file)  
                    file.close()
            
            
            ######################################################################
            # 2)read train sets and trains&save SVM model
            ######################################################################
            else:
                print("terzo ramo")
                
                if rounding is None:
                    print('Training with given data')
                    n_trainSets = (glob.glob(img_path + '*.sav'))
                elif rounding:
                    print('Training with round')
                    n_trainSets = (glob.glob(img_path + '?????_round.sav'))
                else:
                    print('Training with int (i.e. floor)')
                    n_trainSets = (glob.glob(img_path + '?????.sav'))
                xTrain = []
                labels = []
                for i in range(len(n_trainSets)):
                    print(i)
                    try:
                        file = open(n_trainSets[i], 'rb')
                        print(file)
                        xTrain.append(pickle.load(file))
                        labels.append(pickle.load(file))            
                        file.close()
                        print(" Train set loaded")        
                    except:
                        raise IOError("Unable to load train set!")
                
                print(xTrain)
                print(labels)
                
                labels = np.concatenate( labels, axis=0 )
                print(labels)
                
                if ts:
                    minValues = []
                    maxValues = []
                    for i,data in enumerate(xTrain):
                        minValues.append(np.mean(data,0))
                        maxValues.append(np.std(data,0))
                        xTrain[i] = (data - minValues[i] )/maxValues[i]
                xTrain = np.concatenate( xTrain, axis=0 )
                n_classes = np.unique(labels)
                
                try:
                    model = pickle.load(open(output_name, 'rb'))
                    print("Loaded model from disk")        
        
                except IOError:
                        from sklearn.model_selection import StratifiedShuffleSplit
                        n_train = len(xTrain)
                        # Stratified split instead of permutation
                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1223334444)
                        train_index, _ = next(sss.split(xTrain, labels))
                        xTrain = xTrain[train_index]
                        labels = labels[train_index]
                        startTime = datetime.now()
                        if std:
                            print('Normalization by standardization')
                            if ts:
                                print('Tile specific normalization')
                                model, _, _ = defineTrainSVM_std( xTrain, labels, ts)
                            else:
                                model, minValues, maxValues = defineTrainSVM_std( xTrain, labels, ts)
                        else:
                            print('Normalization by min-max scaling')
                            model, minValues, maxValues = defineTrainSVM( xTrain, labels)
                        print('execution_time:',datetime.now() - startTime)
                        print(f'Number of SVs: {len(model.support_vectors_)}')
                        file = open(output_name, 'wb')
                        pickle.dump(model, file)
                        pickle.dump(minValues, file)  
                        pickle.dump(maxValues, file)  
                        pickle.dump(n_classes, file)
                        pickle.dump(xTrain, file)
                        pickle.dump(labels, file)             
                        file.close()
                        print("Model saved")        


def train_main(img_path, output_name, shapefile, GLCM, texture_flag, validate_name):


       ######################################################################
       # compute GLCM features
       ######################################################################
       
        if GLCM: 
            n_imgs = readImgsPaths(img_path)    
            n_imgs.sort(key=my_sorter)
            n_imgs = buildFullPath(img_path,n_imgs)

            computeGLCM(n_imgs[0], img_path) 
       
       ######################################################################
       # 1) read train set points and save it
       ######################################################################
        elif shapefile:  
            print(shapefile,img_path)
            n_imgs = readImgsPaths(img_path)    
            n_imgs.sort(key=my_sorter)
            n_imgs = buildFullPath(img_path,n_imgs)

            print('n_imgs', n_imgs)
            trainingSetPoints = readTrainingSetFromShp2(shapefile, n_imgs[0])
            trainingSetPoints = trainingSetPoints[~np.all(trainingSetPoints == 0, axis=1)] #removes all the elements equal to 0
            if len(trainingSetPoints) !=0:
                xTrain, labels = readTSforTrainingSet(n_imgs, trainingSetPoints)
                labels = labels[~np.all(xTrain == 0, axis=1)] 
                trainingSetPoints = trainingSetPoints[~np.all(xTrain == 0, axis=1)]
                xTrain = xTrain[~np.all(xTrain == 0, axis=1)] #removes all the elements equal to 0

                if texture_flag == 1:
                    xTrain = readTexture(n_imgs, xTrain, trainingSetPoints)
                    
                #######################################################################################################  
                # split PoliMi (Gorica) training set into two: half samples for training and other half for validation   
                #######################################################################################################
                if validate_name: 
                    train = splitSamplesIntoTrainValidation(xTrain, labels, validate_name)                   
                    labels = train[:,0]
                    xTrain = train[:,1:]

                # save the training set
                file = open(output_name, 'wb')
                pickle.dump(xTrain, file)
                pickle.dump(labels, file)  
                file.close()
        
        
        ######################################################################
        # 2)read train sets and trains&save SVM model
        ######################################################################
        else:
            n_trainSets = (glob.glob(img_path + '*.sav'))
            xTrain = []
            labels = []
            
            for i in range(len(n_trainSets)):
                try:
                    file = open(n_trainSets[i], 'rb')
                    xTrain.append(pickle.load(file))
                    labels.append(pickle.load(file))            
                    file.close()
                    print(" Train set loaded")        
                except:
                    raise IOError("Unable to load train set!")
            
            labels = np.concatenate( labels, axis=0 )
            xTrain = np.concatenate( xTrain, axis=0 )
            #xTrain= xTrain[:,:16]

            n_classes = np.unique(labels)
            
            try:
                model = pickle.load(open(output_name, 'rb'))
                print("Loaded model from disk")        
    
            except IOError:
                    #model, minValues, maxValues = defineTrainSVM( xTrain, labels)
                    model, minValues, maxValues = defineTrainSVMparallel( xTrain, labels)
                    
                    file = open(output_name, 'wb')
                    pickle.dump(model, file)
                    pickle.dump(minValues, file)  
                    pickle.dump(maxValues, file)  
                    pickle.dump(n_classes, file)
                    pickle.dump(xTrain, file)
                    pickle.dump(labels, file)             
                    file.close()
                    print("Model saved")        
