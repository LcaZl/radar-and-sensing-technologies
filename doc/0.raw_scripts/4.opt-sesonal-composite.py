# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:57:24 2021

@author: rslab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:29:21 2021

@author: rslab
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:44:19 2020

@author: rslab
"""


import gdal
import numpy as np
import os
import sys
import getopt
import glob
from scipy import ndimage
import math
import time
from pathlib import Path
import multiprocessing as mp
import itertools
from multiprocessing import Pool, cpu_count
import json
import logging
import pathlib
from datetime import datetime as dt
import ntpath
from datetime import datetime
import re

##############################################################################
# The band order of the output Sentinel-2 composites:
#    'B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12'

# The band order of the output Landsat 5/7 composites:
#    'band1','band2','band3','band4','band5','band7'

# The band order of the output Landsat 8 composites:
#    'band2','band3','band4','band5','band6', 'band7'
##############################################################################

def creategeotiff(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], array.shape[2], dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection()) 

    for i in range(0, array.shape[2]):
        dataSet.GetRasterBand(i+1).WriteArray(array[:, :, i])
    dataSet.FlushCache()
    dataSet = None
    del dataSet

    return name


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


def loadTimeSeries3D(n_imgs,pix_region, patch_size, cloudMasks, shadowMasks, nb):
    ts_3D = np.zeros((patch_size* patch_size, len(n_imgs), nb))
    for i in range(len(n_imgs)):   
            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)          
            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            cloud_shadow_mask = (cmask + smask)
            not_cloud_shadow_mask = cloud_shadow_mask^1
            
            img_temp_ds = gdal.Open(n_imgs[i])
            img_temp= np.zeros((patch_size, patch_size, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
                    
            img_temp[img_temp==0]=['nan']       
            img_temp =  np.reshape(img_temp, (patch_size*patch_size, nb))
           
            ts_3D[:, i, : ] = img_temp
           # median = np.nanmedian(ts_3D, axis = 1)
           
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    del cmask_ds
    del smask_ds
    del img_temp_ds
    
    return ts_3D


def loadTimeSeriesMedian3D(n_imgs,pix_region, patch_size, cloudMasks, shadowMasks, nb):
    ts_3D = np.zeros((patch_size* patch_size, len(n_imgs), nb))
    for i in range(len(n_imgs)):   
            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)          
            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            cloud_shadow_mask = (cmask + smask)
            not_cloud_shadow_mask = cloud_shadow_mask^1
            
            img_temp_ds = gdal.Open(n_imgs[i])
            img_temp= np.zeros((patch_size, patch_size, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
                    
            img_temp[img_temp==0]=['nan']       
            img_temp =  np.reshape(img_temp, (patch_size*patch_size, nb))
           
            ts_3D[:, i, : ] = img_temp
            median = np.nanmedian(ts_3D, axis = 1)
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    del cmask_ds
    del smask_ds
    del img_temp_ds
    
    return median


def getSiberiaCompositeMedianaS2Parralel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all = sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))
    months = []
    months.append(['07', '08'])

    cumulativeCloudMask = np.zeros(((10980, 10980)))
    for s in range(len(months)):   

        month = months[s]
        cloudMasks = sorted([x for x in cloudMasks_all if year+month[0] in x or year+month[1] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year+month[0] in x or year+month[1] in x ])
        landsat_img = sorted([x for x in landsat_img_all if year+month[0] in x or year+month[1] in x ])
    
        
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")


        if len(cloudMasks) != 0:
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(landsat_img[0])
            nl=np.int(target_ds.RasterYSize)
            nc=np.int(target_ds.RasterXSize)
            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((nl, nc, nb)))
        

            values=[]
            patch_sizeX = 1098  #2196
            patch_sizeY = 1098  #2196             
            for i in range(10):
                for j in range(10):
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        values.append((landsat_img, pix_region, patch_sizeX, cloudMasks, shadowMasks, nb))              
            values = tuple(values)
            
            
            p = Pool(nbOfProcesses)
            result = p.starmap(loadTimeSeriesMedian3DSiberia, values)
           # p.close()
           ## p.join()
            k=0
            for i in range(10):
                for j in range(10):
                    temp = result[k]
                    sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp[0], [patch_sizeX,patch_sizeY, nb])
                    cumulativeCloudMask[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  temp[1]
                    k = k+1           
           
            creategeotiff(os.path.join(str(output_path),'yearlyCompositeMedianMaskedParaler_' + tile + year + '_' + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            outputName = 'cumulativeCloudMask_' + str(tile) + str(year) + '_'  + '.tif'        
            creategeotiffOneBand(outputName, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
            target_ds = None
            del target_ds
            cmask_ds = None
            del cmask_ds




def loadTimeSeriesMedian3DSiberia(n_imgs,pix_region, patch_size, cloudMasks, shadowMasks, nb):
    ts_3D = np.zeros((patch_size* patch_size, len(n_imgs), nb))
    cumulativeCloudMask = np.zeros(((patch_size, patch_size))) 
    for i in range(len(n_imgs)):   
            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)          
            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            cloud_shadow_mask = (cmask + smask)
            not_cloud_shadow_mask = cloud_shadow_mask^1
            
            img_temp_ds = gdal.Open(n_imgs[i])
            img_temp= np.zeros((patch_size, patch_size, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
              
            noData= np.count_nonzero(img_temp, axis=2)
            noData[noData>0] =1
            cumulativeCloudMask = cumulativeCloudMask + noData
            
            img_temp[img_temp==0]=['nan']       
            img_temp =  np.reshape(img_temp, (patch_size*patch_size, nb))      
            ts_3D[:, i, : ] = img_temp
            median = np.nanmedian(ts_3D, axis = 1)
        
        
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    del cmask_ds
    del smask_ds
    del img_temp_ds
    
    return median, cumulativeCloudMask



def getSiberiaCompositeMedianaS2(input_path, output_path, cloudPath , shadowPath, year ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))
    months = []
    months.append(['07', '08'])

    for s in range(len(months)):   

        month = months[s]
        cloudMasks = sorted([x for x in cloudMasks_all if year+month[0] in x or year+month[1] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year+month[0] in x or year+month[1] in x ])
        landsat_img = sorted([x for x in landsat_img_all if year+month[0] in x or year+month[1] in x ])
    
        
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")


        if len(cloudMasks) != 0:
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(landsat_img[0])
            nl=np.int(target_ds.RasterYSize)
            nc=np.int(target_ds.RasterXSize)
            nb = target_ds.RasterCount                  
        
            sesonalComposite = np.zeros(((nl, nc, nb)))
            cumulativeCloudMask = np.zeros((nl, nc))

            for i in range(1):

                patch_sizeX = 2196 
                patch_sizeY = 2196                

                for i in range(5):
                    for j in range(5):
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        img3d, cumulativeCloudMaskTemp = loadTimeSeriesMedian3DSiberia(landsat_img, pix_region, patch_sizeX, cloudMasks, shadowMasks, nb)
                        #temp = np.nanmedian(img3d, axis = 1)
                        sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(img3d ,[patch_sizeX,patch_sizeY, nb])
                        cumulativeCloudMask[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  cumulativeCloudMaskTemp
                       
            creategeotiff(os.path.join(str(output_path),'yearlyCompositeMedianMasked_'+ str(tile) + str(year) + '_' + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            outputName = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + '_' + str(s+1) + '.tif')      
            creategeotiffOneBand(outputName, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)     
 
            target_ds = None
            del target_ds
            cmask_ds = None
            del cmask_ds



def getImagesInBetween(images, after, before, year):
    imagesInBetween = []
    for z in range(len(images)):
            img = os.path.basename(images[z])
            img_date = img[img.index(year):img.index(year)+8]
            img_date = datetime.strptime(img_date, "%Y%m%d")         
            if after < img_date < before:
                imagesInBetween.append(images[z])
                
    return imagesInBetween


def loadTimeSeriesMedian3DCumulative(n_imgs,pix_region, patch_size, cloudMasks, shadowMasks, nb):
    
    ts_3D = np.zeros((patch_size* patch_size, len(n_imgs), nb))
    cumulativeCloudMask = np.zeros(((patch_size, patch_size))) 
    for i in range(len(n_imgs)):   
            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)          
            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            cloud_shadow_mask = (cmask + smask)
            not_cloud_shadow_mask = cloud_shadow_mask^1
            
            img_temp_ds = gdal.Open(n_imgs[i])
            img_temp = np.zeros((patch_size, patch_size, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
                    
            noData= np.count_nonzero(img_temp, axis=2)
            noData[noData>0] =1
            cumulativeCloudMask = cumulativeCloudMask + noData
                    
            img_temp[img_temp==0]=['nan']       
            img_temp =  np.reshape(img_temp, (patch_size*patch_size, nb))
            ts_3D[:, i, : ] = img_temp
            median = np.nanmedian(ts_3D, axis = 1)
            
    cumulativeCloudMask[cumulativeCloudMask<3] = 0
    cumulativeCloudMask[cumulativeCloudMask!=0] = 1
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    del cmask_ds
    del smask_ds
    del img_temp_ds
    
    return median, cumulativeCloudMask



def getMontlyCompositeMedianaS2Parralel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))   
    
    if len(cloudMasks_all) != len(landsat_img_all) or len(shadowMasks_all) != len(landsat_img_all):
        raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")
    
    months = []   
    months.append(['0101', '0215'])
    months.append(['0115', '0315'])
    months.append(['0215', '0415'])
    months.append(['0315', '0515'])
    months.append(['0415', '0615'])
    months.append(['0515', '0715'])
    months.append(['0615', '0815'])
    months.append(['0715', '0915'])
    months.append(['0815', '1015'])
    months.append(['0915', '1115'])
    months.append(['1015', '1215'])
    months.append(['1115', '1230'])
    
    cumulativeCloudMask = np.zeros(((10980, 10980)))
    
    for s in range(12):   
        month = months[s] 
        after = datetime.strptime(year+month[0], "%Y%m%d")
        before = datetime.strptime(year+month[1], "%Y%m%d")
        landsat_img = sorted(getImagesInBetween(landsat_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))

          
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")


        if len(cloudMasks) != 0:
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(landsat_img[0])
            nl=np.int(target_ds.RasterYSize)
            nc=np.int(target_ds.RasterXSize)
            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((nl, nc, nb)))
           
            values=[]
            patch_sizeX = 1098  #1098 2196
            patch_sizeY = 1098  #1098  2196             
            for i in range(10):   
                for j in range(10):
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        values.append((landsat_img, pix_region, patch_sizeX, cloudMasks, shadowMasks, nb))              
            values = tuple(values)
            
            
            p = Pool(nbOfProcesses)
            result = p.starmap(loadTimeSeriesMedian3DCumulative, values)
           # p.close()
           ## p.join()
            k=0
            for i in range(10): 
                for j in range(10):            
                    temp = result[k]
                    sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp[0], [patch_sizeX,patch_sizeY, nb])
                    cumulativeCloudMask[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  temp[1]
                    k = k+1                         
                        
            creategeotiff(os.path.join(str(output_path),'montlyCompositeMedianMasked_' + str(tile) + str(year) + '_'  + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            outputName = os.path.join(str(output_path),'montlycumulativeCloudMask_' + str(tile) + str(year) + '_'  + str(s+1) + '.tif' )       
            creategeotiffOneBand(outputName, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
   
            target_ds = None
            del target_ds
            cmask_ds = None
            del cmask_ds
                        
        if s == 11:  # after december has been generated merge all masks together
            start = str(tile) + str(year) + '_'
            end = '.tif'
            cumulativeCloudMaskUnited = np.zeros(((10980, 10980)))
            cumulativeMasks_all =  sorted(glob.glob(os.path.join(str(output_path),'montlycumulativeCloudMask_*.tif')))   
            months_used = ''
            for p in range(len(cumulativeMasks_all)):
                cumulativeCloudMaskTemp_ds = gdal.Open(cumulativeMasks_all[p])
                cumulativeCloudMaskTemp = cumulativeCloudMaskTemp_ds.ReadAsArray(0, 0, np.int(nc),np.int(nl))
                cumulativeCloudMaskUnited  = cumulativeCloudMaskUnited + cumulativeCloudMaskTemp
                            
                months_used = months_used + '_' + re.search('%s(.*)%s' % (start, end),cumulativeMasks_all[p]).group(1)
                
            outputName = os.path.join(str(output_path),'montlycumulativeCloudMask_' + str(tile) + str(year) + str(months_used) + '.tif' )       
            creategeotiffOneBand(outputName, cumulativeCloudMaskUnited, 0, cumulativeCloudMaskTemp_ds,  gdal.GDT_Byte)   
            cumulativeCloudMaskTemp_ds = None
            del cumulativeCloudMaskTemp_ds


def getMontlyCompositeMedianaS2(input_path, output_path, cloudPath , shadowPath, year ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))   
    
    months = []
    months.append(['0101', '0215'])
    months.append(['0115', '0315'])
    months.append(['0215', '0415'])
    months.append(['0315', '0515'])
    months.append(['0415', '0615'])
    months.append(['0515', '0715'])
    months.append(['0615', '0815'])
    months.append(['0715', '0915'])
    months.append(['0815', '1015'])
    months.append(['0915', '1115'])
    months.append(['1015', '1215'])
    months.append(['1115', '1230'])

      
    for s in range(12):   
        month = months[s] 
        after = datetime.strptime(year+month[0], "%Y%m%d")
        before = datetime.strptime(year+month[1], "%Y%m%d")
        landsat_img = sorted(getImagesInBetween(landsat_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))

          
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")
  

        if len(cloudMasks) != 0:
            target_ds = gdal.Open(landsat_img[0])
            nl=np.int(target_ds.RasterYSize)
            nc=np.int(target_ds.RasterXSize)
            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((nl, nc, nb)))

            for i in range(1): 

                patch_sizeX = 2196 
                patch_sizeY = 2196                

                for i in range(5): 
                    for j in range(5): 
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        img3d = loadTimeSeries3D(landsat_img, pix_region, patch_sizeX, cloudMasks, shadowMasks, nb)
                        temp = np.nanmedian(img3d, axis = 1)
                        #tempMin = np.nanmin(img3d, axis=1)
                       
                        sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp ,[patch_sizeX,patch_sizeY, nb])
                        
            creategeotiff(os.path.join(str(output_path),'montlyCompositeMedianMasked_' + tile + year + '_' + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            target_ds = None
            del target_ds


            
def clipByExtent(sesonalComposite, seasonal_ds, img_temp, img_temp_ds):

    gt1 = seasonal_ds.GetGeoTransform()
    gt2 = img_temp_ds.GetGeoTransform()
    nb = seasonal_ds.RasterCount
    if gt1[0] < gt2[0]: #CONDITIONAL TO SELECT THE CORRECT ORIGIN
        gt3 = gt2[0]
    else:
        gt3 = gt1[0]
    if gt1[3] < gt2[3]:
        gt4 = gt1[3]
    else:
        gt4 = gt2[3]
        
    xOrigin = gt3
    yOrigin = gt4
    pixelWidth = gt1[1]
    pixelHeight = gt1[5]

    r1 = [gt1[0], gt1[3],gt1[0] + (gt1[1] * seasonal_ds.RasterXSize), gt1[3] + (gt1[5] * seasonal_ds.RasterYSize)]
    r2 = [gt2[0], gt2[3],gt2[0] + (gt2[1] * img_temp_ds.RasterXSize), gt2[3] + (gt2[5] * img_temp_ds.RasterYSize)]
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]

    xmin = intersection[0]
    xmax = intersection[2]
    ymin = intersection[3]
    ymax = intersection[1]

    
    xoff_1 = int((xOrigin - gt1[0] )/pixelWidth)
    yoff_1 = int((gt1[3]-yOrigin)/pixelWidth)
    xoff_2 = int((xOrigin - gt2[0] )/pixelWidth)
    yoff_2 = int((gt2[3]-yOrigin)/pixelWidth)
    
    xcount = int((xmax - xmin)/pixelWidth)
    ycount = int((ymax - ymin)/pixelWidth)

    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, nb, gdal.GDT_UInt16)
    target_ds.SetGeoTransform((xOrigin, pixelWidth, 0, yOrigin, 0, pixelHeight))
    target_ds.SetProjection(seasonal_ds.GetProjection())
    

    return xoff_1, yoff_1, xoff_2, yoff_2, xcount, ycount, target_ds
            

def alignRemainingImg(aligned, yoff_1, xoff_1, ycount, xcount ):
    for i in range(len(aligned)):
        temp = aligned[i] 
        aligned[i] = temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount,:]
        
    return aligned


def alignLandsatComposites(landsat_img):
    aligned = []
    target_ds = gdal.Open(landsat_img[0])
    img_temp = target_ds.ReadAsArray(0, 0, np.int(target_ds.RasterXSize),np.int(target_ds.RasterYSize))
    img_temp = np.moveaxis(img_temp, 0, -1)

    for k in range(len(landsat_img)-1):
        
        img_temp_ds1 = gdal.Open(landsat_img[k+1])
        img_temp1 = img_temp_ds1.ReadAsArray(0, 0, np.int(img_temp_ds1.RasterXSize), np.int(img_temp_ds1.RasterYSize))  
        img_temp1 = np.moveaxis(img_temp1, 0, -1)
           
        yoff_1, xoff_1, yoff_2, xoff_2, ycount, xcount, target_ds = clipByExtent(img_temp, target_ds, img_temp1, img_temp_ds1)
        
        img_temp = img_temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount,:]
        img_temp1 = img_temp1[xoff_2:xoff_2+xcount,yoff_2:yoff_2+ycount,:]
        
        if k==0:
           # img_temp = img_temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount,:]
            aligned.append(img_temp)
        else:
            aligned = alignRemainingImg(aligned, yoff_1, xoff_1, ycount, xcount )
            
        aligned.append(img_temp1)
        
    for k in range(len(landsat_img)):         
        base = os.path.basename(landsat_img[k])
        directory = os.path.dirname(landsat_img[k])
        new_directory = os.path.join(directory, 'coregistered')
        Path(new_directory).mkdir(parents=True, exist_ok=True)
        output_name = os.path.join(new_directory, base[:-4] + '_C' +'.tif')
        creategeotiff(output_name, aligned[k], 0, target_ds,  gdal.GDT_UInt16)
        
    img_temp = None
    target_ds = None
    img_temp1 = None
    img_temp_ds1 = None  
    del target_ds
    del img_temp_ds1
        

def alignRemainingImg1band(aligned, yoff_1, xoff_1, ycount, xcount ):
    
    for i in range(len(aligned)):
        temp = aligned[i] 
        aligned[i] = temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount]
        
    return aligned


def alignLandsatComposites1band(landsat_img):
    aligned = []
    target_ds = gdal.Open(landsat_img[0])
    img_temp = target_ds.ReadAsArray(0, 0, np.int(target_ds.RasterXSize),np.int(target_ds.RasterYSize))
    
    for k in range(len(landsat_img)-1):
        
        img_temp_ds1 = gdal.Open(landsat_img[k+1])
        img_temp1 = img_temp_ds1.ReadAsArray(0, 0, np.int(img_temp_ds1.RasterXSize), np.int(img_temp_ds1.RasterYSize))  
                   
        yoff_1, xoff_1, yoff_2, xoff_2, ycount, xcount, target_ds = clipByExtent(img_temp, target_ds, img_temp1, img_temp_ds1)
        
        img_temp = img_temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount]
        img_temp1 = img_temp1[xoff_2:xoff_2+xcount,yoff_2:yoff_2+ycount]
        
        if k==0:
            #img_temp = img_temp[xoff_1:xoff_1+xcount,yoff_1:yoff_1+ycount]
            aligned.append(img_temp)
        else:
            aligned = alignRemainingImg1band(aligned, yoff_1, xoff_1, ycount, xcount )
            
        aligned.append(img_temp1)
        
    for k in range(len(landsat_img)):         
        base = os.path.basename(landsat_img[k])
        directory = os.path.dirname(landsat_img[k])
        new_directory = os.path.join(directory, 'coregistered')
        Path(new_directory).mkdir(parents=True, exist_ok=True)
        output_name = os.path.join(new_directory, base[:-4] + '_C' +'.tif')
   
        # name = landsat_img[k]
        # name = name[:-4] + '_C' +'.tif'
        creategeotiffOneBand(output_name, aligned[k], 0, target_ds,  gdal.GDT_Byte)
    #return aligned, target_ds
        
    img_temp = None
    target_ds = None
    img_temp1 = None
    img_temp_ds1 = None
    del target_ds
    del img_temp_ds1
    
    
def coregisterImages(input_path, cloudPath , shadowPath, area, path, row, year):
    p = path + '0' + row + year

    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*' + p + '*cloudMediumMask.tif')))
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath,'*' + p + '*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*' + p + '*.tif')))
    
    alignLandsatComposites(landsat_img_all)
    alignLandsatComposites1band(shadowMasks_all)
    alignLandsatComposites1band(cloudMasks_all)
    
    if area == "Siberia" or area == "siberia":
        cloudMasks_all =  (glob.glob(os.path.join(cloudPath, '*' + p + '*snowIceMask.tif')))  
        alignLandsatComposites1band(cloudMasks_all)
        
        
        

def coregisterImagesSiberia(input_path, cloudPath , shadowPath, area, path, row, year):
    #p = path + '0' + row + year

    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath,'*shadowMask.tif')))   
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))
    
    alignLandsatComposites(landsat_img_all)
    alignLandsatComposites1band(shadowMasks_all)
    alignLandsatComposites1band(cloudMasks_all)
    
    if area == "Siberia" or area == "siberia":
        cloudMasks_all =  (glob.glob(os.path.join(cloudPath, '*snowIceMask.tif')))  
        alignLandsatComposites1band(cloudMasks_all)
    
    
    
    
def getSesonalCompositeMedianaLandsat(input_path, output_path, tile, year, cloudPath , shadowPath ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask*.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask*.tif')))
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path,  '*.tif')))
    months = []
    months.append(['01', '02', '03']) # jan feb march
    months.append(['04', '05', '06']) # april may ...
    months.append(['07', '08', '09'])
    months.append(['10', '11', '12'])

    for s in range(4): 
        
        month = months[s]     
        landsat_img = []
        shadowMasks = []
        cloudMasks = []
        
        #divide images by seasons
        for i in range(len(landsat_img_all)):            
            temp= ntpath.basename(landsat_img_all[i])[21:23]
            if temp in month:
                landsat_img.append(landsat_img_all[i])
                shadowMasks.append(shadowMasks_all[i])
                cloudMasks.append(cloudMasks_all[i])

        landsat_img = sorted(landsat_img)
        shadowMasks = sorted(shadowMasks)
        cloudMasks = sorted(cloudMasks)
     
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

        if len(cloudMasks) != 0:           
            target_ds = gdal.Open(landsat_img[0])
            xsize=np.int(target_ds.RasterXSize) 
            ysize =np.int(target_ds.RasterYSize)
        
            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((ysize, xsize, nb)))
            
            for i in range(len(landsat_img)):
                patchSizeY = math.floor(ysize/5)   
                patchSizeX = math.floor(xsize/5)      
                patchSizeXorig = math.floor(xsize/5)
                patchSizeYorig = math.floor(ysize/5)   

                for i in range(5):
                    for j in range(5):

                       if i==4 and patchSizeX*5 != xsize:
                           patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig-1
                       else: 
                           patchSizeX = math.floor(xsize/5)
        
                       if j==4 and patchSizeY*5 != ysize:
                           patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig-1
                       else: 
                           patchSizeY = math.floor(ysize/5)
                           
                       pix_region = np.array([i*patchSizeXorig, j*patchSizeYorig])
                       img3d = loadTimeSeries3DLandsat(landsat_img, pix_region,  patchSizeX, patchSizeY, cloudMasks, shadowMasks, nb)

                       temp = np.nanmedian(img3d, axis = 1)
                       sesonalComposite[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX, :] =  np.reshape(temp ,[patchSizeY,patchSizeX, nb])

  
            creategeotiff(os.path.join(str(output_path),'sesonalCompositeMedianMasked_' + str(tile) + year +'_' + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)   

            target_ds = None
            del target_ds            

            
   
def getSesonalCompositeMedianaLandsatSiberia(input_path, output_path,  tile,  year, cloudPath, shadowPath ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks =   sorted(glob.glob(os.path.join(cloudPath,  '*cloudMediumMask*.tif')))    
    shadowMasks =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask*.tif')))   
    snowMasks = sorted(glob.glob(os.path.join(cloudPath,  '*snowIceMask*.tif')))   
    landsat_img = sorted(glob.glob(os.path.join(input_path, '*.tif')))
    
    #months = []
    #months.append(['04', '05', '06', '07', '08', '09']) # jan feb march
    for s in range(1): 
                
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

        if len(cloudMasks) != 0:            
            target_ds = gdal.Open(landsat_img[0])
            xsize=np.int(target_ds.RasterXSize)
            ysize =np.int(target_ds.RasterYSize)

            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((ysize, xsize, nb)))
            
            for i in range(1):  #(len(landsat_img)):
                patchSizeY = math.floor(ysize/5)   
                patchSizeX = math.floor(xsize/5)      
                patchSizeXorig = math.floor(xsize/5)
                patchSizeYorig = math.floor(ysize/5)   

                for i in range(5):
                    for j in range(5):

                       if i==4 and patchSizeX*5 != xsize:
                           patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig-1
                       else: 
                           patchSizeX = math.floor(xsize/5)
        
                       if j==4 and patchSizeY*5 != ysize:
                           patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig-1
                       else: 
                           patchSizeY = math.floor(ysize/5)
                           
                       pix_region = np.array([i*patchSizeXorig, j*patchSizeYorig])
                       img3d, img3d2 = loadTimeSeries3DLandsatSiberia(landsat_img, pix_region,  patchSizeX, patchSizeY, cloudMasks, shadowMasks, snowMasks, nb)

                       temp = np.nanmedian(img3d, axis = 1)
                       is_all_zero = np.all(np.isnan(temp), axis = 1)
                       #in case there is no data fill with minimum
                       minTable = np.min(img3d2, axis = 1)
                       temp[is_all_zero,:] = minTable[is_all_zero,:]
                       #border = np.all(temp> 55535, axis = 1)
                       #temp[border,:] = 0

                       sesonalComposite[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX, :] =  np.reshape(temp ,[patchSizeY,patchSizeX, nb])
           

            creategeotiff(os.path.join(str(output_path),'yearlyCompositeMedianMasked_' + str(tile) + year + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            target_ds = None
            del target_ds



def loadTimeSeries3DLandsat(landsat_img,pix_region, patchSizeX, patchSizeY, cloudMasks, shadowMasks, nb):
    ts_3D = np.zeros((patchSizeX*patchSizeY, len(landsat_img), nb))
    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)

            cloud_shadow_mask = (cmask + smask)
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patchSizeY, patchSizeX, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)    
                   # not_cloud_shadow_mask[img_temp[:,:,b] > 55535] = 0  # removes Landsat 7 stripes 
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
                    
                    
            img_temp[img_temp==0]=['nan']
            img_temp =  np.reshape(img_temp, (patchSizeX*patchSizeY, nb))
           
            ts_3D[:, i, : ] = img_temp
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None 
    del cmask_ds
    del smask_ds
    del img_temp_ds

    return ts_3D


def loadTimeSeries3DLandsatSiberia(landsat_img,pix_region, patchSizeX, patchSizeY, cloudMasks, shadowMasks, snowMasksList, nb):
    ts_3D = np.zeros((patchSizeX*patchSizeY, len(landsat_img), nb))
    ts_3D2 = np.zeros((patchSizeX*patchSizeY, len(landsat_img), nb))

    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)
            
            snowMasks_ds = gdal.Open(snowMasksList[i])
            snowMasks = snowMasks_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)

            cloud_shadow_mask = (cmask + smask + snowMasks)
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patchSizeY, patchSizeX, nb))
            img_tempNoMask= np.zeros((patchSizeY, patchSizeX, nb))

            for b in range(nb):
                    img_tempNoMask[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)    
                   # img_tempNoMask[(img_tempNoMask[:,:,b] > 55535) & (img_tempNoMask[:,:,b] < 55550)] = 0  #strips for L7 and Landsat borders
                   # not_cloud_shadow_mask[img_tempNoMask[:,:,b] > 55535] = 0  # removes Landsat 7 stripes 
                    img_temp[:,:,b] = np.multiply(img_tempNoMask[:,:,b], not_cloud_shadow_mask)
                    
                    
            img_temp[img_temp==0]=['nan']
            img_temp =  np.reshape(img_temp, (patchSizeX*patchSizeY, nb))
            img_tempNoMask =  np.reshape(img_tempNoMask, (patchSizeX*patchSizeY, nb))
           
            ts_3D[:, i, : ] = img_temp
            ts_3D2[:, i, : ] = img_tempNoMask
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None 
    del cmask_ds
    del smask_ds
    del img_temp_ds

    return ts_3D, ts_3D2



def loadCumulativeCloudMask(landsat_img, pix_region, patchSizeX, patchSizeY, cloudMasks, shadowMasks, snowMasks=""):
    cumulativeCloudMask = np.zeros((patchSizeX,patchSizeY))

    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)
            
            if len(snowMasks) != 0:
                snowMasks_ds = gdal.Open(shadowMasks[i])
                snowMasks = snowMasks_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)
                cloud_shadow_mask = cmask + smask  + snowMasks
            else:
                cloud_shadow_mask = (cmask + smask ) 
                
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patchSizeY, patchSizeX))
            img_tempNoMask= np.zeros((patchSizeY, patchSizeX))

            for b in range(1):
                    img_tempNoMask[:, :] = img_temp_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patchSizeX, patchSizeY)    
                    img_temp[:,:] = np.multiply(img_tempNoMask[:,:], not_cloud_shadow_mask)
                                        
            img_temp[img_temp!=0]=1
            img_temp =  np.reshape(img_temp, (patchSizeX,patchSizeY))
            cumulativeCloudMask = cumulativeCloudMask + img_temp
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None 
    del cmask_ds
    del smask_ds
    del img_temp_ds

    return cumulativeCloudMask 



def computeNDVI(img_path, cloudPath, output_path):    
    landsat_img_all =  (glob.glob(os.path.join(img_path, '*.tif')))
    cloudMasks_all =  (glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   

    for i in range(len(landsat_img_all)):
          img_ds = gdal.Open(landsat_img_all[i])          
          cloud_ds = gdal.Open(cloudMasks_all[i])            
          cloud = cloud_ds.GetRasterBand(1).ReadAsArray(0, 0, np.int(cloud_ds.RasterXSize), np.int(cloud_ds.RasterYSize))
          not_cloud_mask = cloud^1

          band4 = (img_ds.GetRasterBand(4).ReadAsArray(0, 0, np.int(img_ds.RasterXSize), np.int(img_ds.RasterYSize))).astype(np.float)
          band3 = (img_ds.GetRasterBand(3).ReadAsArray(0, 0, np.int(img_ds.RasterXSize), np.int(img_ds.RasterYSize))).astype(np.float)
          ndvi = (band4-band3)/(band4+band3 + 0.000001)
          ndvi = ndvi*not_cloud_mask
          ndvi[ndvi==0] = np.nan  
          ndvi[ndvi>1] = 1  
          ndvi[ndvi<-1] = -1  
          
          filename = os.path.basename(landsat_img_all[i])
          name = os.path.join(output_path, filename[:-4] + '_NDVI.tif')
          creategeotiffOneBand(name, ndvi,  0, img_ds,  gdal.GDT_Float32)
          
    cloud = None
    cloud_ds = None
    del cloud
    del cloud_ds
    
    
def computeNDVISiberia(input_path, cloudPath, shadowPath, output_path): 
    landsat_img_all =  (glob.glob(os.path.join(input_path, '*.tif')))
    cloudMasks_all =  (glob.glob(os.path.join(cloudPath, '*cloudMediumMask*.tif')))   
    shadowMasks_all =  (glob.glob(os.path.join(shadowPath, '*shadow*Mask*.tif')))   
    snowMasks_all =  (glob.glob(os.path.join(cloudPath, '*snowIceMask*.tif')))   

    months = []   
    months.append(['0401', '0630'])
    months.append(['0701', '0930'])
    restoreIx = 2
    
    for s in range(len(months)):   
        month = months[s] 
        after = datetime.strptime(year+month[0], "%Y%m%d")
        before = datetime.strptime(year+month[1], "%Y%m%d")
        landsat_img = sorted(getImagesInBetween(landsat_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))
        snowMasks = sorted(getImagesInBetween(snowMasks_all, after, before, year))

        if len(landsat_img) == 0:
                  restoreIx = s
        else:
                  #first create composite
                  target_ds = gdal.Open(landsat_img[0])
                  xsize=np.int(target_ds.RasterXSize)
                  ysize =np.int(target_ds.RasterYSize)
        
                  nb = target_ds.RasterCount                  
                  sesonalComposite = np.zeros(((ysize, xsize, nb)))
                                   
                  patchSizeY = math.floor(ysize/5)   
                  patchSizeX = math.floor(xsize/5)      
                  patchSizeXorig = math.floor(xsize/5)
                  patchSizeYorig = math.floor(ysize/5)   
        
                  for i in range(5):
                      for j in range(5):
        
                               if i==4 and patchSizeX*5 != xsize:
                                   patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig-1
                               else: 
                                   patchSizeX = math.floor(xsize/5)
                
                               if j==4 and patchSizeY*5 != ysize:
                                   patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig-1
                               else: 
                                   patchSizeY = math.floor(ysize/5)
                                   
                               pix_region = np.array([i*patchSizeXorig, j*patchSizeYorig])
                               img3d, img3d2 = loadTimeSeries3DLandsatSiberia(landsat_img, pix_region,  patchSizeX, patchSizeY, cloudMasks, shadowMasks, snowMasks, nb)
        
                               temp = np.nanmedian(img3d, axis = 1)
                               is_all_zero = np.all(np.isnan(temp), axis = 1)
                               #in case there is no data fill with minimum
                               minTable = np.min(img3d2, axis = 1)
                               temp[is_all_zero,:] = minTable[is_all_zero,:]
        
                               sesonalComposite[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX, :] =  np.reshape(temp ,[patchSizeY,patchSizeX, nb])
                            
                  #compute NDVi
                  band4 = sesonalComposite[:,:,4] 
                  band3 = sesonalComposite[:,:,3]
                  ndvi = (band4-band3)/(band4+band3 + 0.000001)
                  ndvi[ndvi==0] = np.nan  
                  ndvi[ndvi>1] = 1  
                  ndvi[ndvi<-1] = -1  
                  
                  name = os.path.join(output_path,  'NDVI_'+ str(s) +'.tif')
                  creategeotiffOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
    
    #if for one composite no images are avaiable
    if restoreIx  == 0:
        otherNDVIname = os.path.join(output_path,  'NDVI_1.tif')
        target_ds = gdal.Open(otherNDVIname)            
        ndvi = target_ds.GetRasterBand(1).ReadAsArray(0, 0, target_ds.RasterXSize, target_ds.RasterYSize) 
        
        name = os.path.join(output_path,  'NDVI_ '+ str(restoreIx) +'.tif')
        creategeotiffOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
    
    elif restoreIx  == 1:
        otherNDVIname = os.path.join(output_path,  'NDVI_0.tif')
        target_ds = gdal.Open(otherNDVIname)
        ndvi = target_ds.GetRasterBand(1).ReadAsArray(0, 0, target_ds.RasterXSize, target_ds.RasterYSize) 
        name = os.path.join(output_path,  'NDVI_ '+ str(restoreIx) +'.tif')
        creategeotiffOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
        
    #DON'T forget  about NDVI restoration!!!!
    
    cloud = None
    cloud_ds = None
    target_ds = None

    del cloud
    del cloud_ds
    del target_ds


def getCumulativeCloudMask(input_path, cloudPath, shadowPath, output_path, tile, year): 
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask*.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask*.tif')))
    landsat_img_all =  sorted(glob.glob(os.path.join(input_path,  '*.tif')))
    months = []
    months.append(['01', '02', '03']) # jan feb march
    months.append(['04', '05', '06']) # april may ...
    months.append(['07', '08', '09'])
    months.append(['10', '11', '12'])


    cmask_ds = gdal.Open(cloudMasks_all[0])
    nl = np.int(cmask_ds.RasterXSize) 
    nc = np.int(cmask_ds.RasterYSize)
    cumulativeCloudMask = np.zeros(((nc, nl)))
    cumulativeCloudMaskUnited= np.zeros(((nc, nl)))
    
    for s in range(4): 
        
        month = months[s]     
        landsat_img = []
        shadowMasks = []
        cloudMasks = []
        
        #divide images by seasons
        for i in range(len(landsat_img_all)):            
            temp= ntpath.basename(landsat_img_all[i])[21:23]
            if temp in month:
                landsat_img.append(landsat_img_all[i])
                shadowMasks.append(shadowMasks_all[i])
                cloudMasks.append(cloudMasks_all[i])

        landsat_img = sorted(landsat_img)
        shadowMasks = sorted(shadowMasks)
        cloudMasks = sorted(cloudMasks)
     
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

        if len(cloudMasks) != 0:           
    
            target_ds = gdal.Open(cloudMasks[0])
            xsize=np.int(target_ds.RasterXSize)
            ysize =np.int(target_ds.RasterYSize)
            
            nb = target_ds.RasterCount                  
            cumulativeCloudMask = np.zeros((ysize, xsize))
                                       
            patchSizeY = math.floor(ysize/5)   
            patchSizeX = math.floor(xsize/5)      
            patchSizeXorig = math.floor(xsize/5)
            patchSizeYorig = math.floor(ysize/5)   
            
            for i in range(5):
                for j in range(5):
            
                    if i==4 and patchSizeX*5 != xsize:
                        patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig-1
                    else: 
                        patchSizeX = math.floor(xsize/5)
                    
                    if j==4 and patchSizeY*5 != ysize:
                          patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig-1
                    else: 
                          patchSizeY = math.floor(ysize/5)
                                       
                    pix_region = np.array([i*patchSizeXorig, j*patchSizeYorig])
                    cumulativeCloudMask[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX] = loadCumulativeCloudMask(landsat_img, pix_region,  patchSizeX, patchSizeY, cloudMasks, shadowMasks)
           
            name =  os.path.join(output_path,  'cumulativeCloudMask_'+ (tile) + str(year) + '_'+str(s+1)+'.tif')
            creategeotiffOneBand(name, cumulativeCloudMask,  0, target_ds,  gdal.GDT_Float32)
            
        if s == 3:  # after december has been generated merge all masks together
            start = str(tile) + str(year) + '_'
            end = '.tif'
            cumulativeCloudMaskUnited = np.zeros(((nc, nl)))
            cumulativeMasks_all =  sorted(glob.glob(os.path.join(str(output_path),'cumulativeCloudMask_*.tif')))   
            months_used = ''
            for p in range(len(cumulativeMasks_all)):
                cumulativeCloudMaskTemp_ds = gdal.Open(cumulativeMasks_all[p])
                cumulativeCloudMaskTemp = cumulativeCloudMaskTemp_ds.ReadAsArray(0, 0, np.int(nl),np.int(nc))
                cumulativeCloudMaskUnited  = cumulativeCloudMaskUnited + cumulativeCloudMaskTemp
                            
                months_used = months_used + '_' + re.search('%s(.*)%s' % (start, end),cumulativeMasks_all[p]).group(1)
                
            outputName = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + str(months_used) + '.tif' )       
            creategeotiffOneBand(outputName, cumulativeCloudMaskUnited, 0, cumulativeCloudMaskTemp_ds,  gdal.GDT_Byte)   
            cumulativeCloudMaskTemp_ds = None
            del cumulativeCloudMaskTemp_ds
            
    cmask_ds = None
    del cmask_ds
    

          
   
def getCumulativeCloudMaskSiberia(input_path, cloudPath, shadowPath, output_path, tile, year):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks =   sorted(glob.glob(os.path.join(cloudPath,  '*cloudMediumMask*.tif')))    
    shadowMasks =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask*.tif')))   
    snowMasks = sorted(glob.glob(os.path.join(cloudPath,  '*snowIceMask*.tif')))   
    landsat_img = sorted(glob.glob(os.path.join(input_path, '*.tif'))) 

    cmask_ds = gdal.Open(cloudMasks[0])
    nl = np.int(cmask_ds.RasterXSize) 
    nc = np.int(cmask_ds.RasterYSize)
    cumulativeCloudMask = np.zeros(((nc, nl)))

                
    if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
        raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

    if len(cloudMasks) != 0:            
            target_ds = gdal.Open(landsat_img[0])
            xsize=np.int(target_ds.RasterXSize)
            ysize =np.int(target_ds.RasterYSize)
            nb = target_ds.RasterCount                  
            
            patchSizeY = math.floor(ysize/5)   
            patchSizeX = math.floor(xsize/5)      
            patchSizeXorig = math.floor(xsize/5)
            patchSizeYorig = math.floor(ysize/5)   

            for i in range(5):
                for j in range(5):

                       if i==4 and patchSizeX*5 != xsize:
                           patchSizeX = (xsize - patchSizeXorig*5) + patchSizeXorig-1
                       else: 
                           patchSizeX = math.floor(xsize/5)
        
                       if j==4 and patchSizeY*5 != ysize:
                           patchSizeY = (ysize - patchSizeYorig*5) + patchSizeYorig-1
                       else: 
                           patchSizeY = math.floor(ysize/5)
                           
                       pix_region = np.array([i*patchSizeXorig, j*patchSizeYorig])

                       cumulativeCloudMask[j*patchSizeYorig:j*patchSizeYorig+patchSizeY, i*patchSizeXorig:i*patchSizeXorig+patchSizeX] =  loadCumulativeCloudMask(landsat_img, pix_region,  patchSizeX, patchSizeY, cloudMasks, shadowMasks, snowMasks)
           
            
    outputName = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + '.tif')
    creategeotiffOneBand(outputName, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
    cmask_ds = None
    del cmask_ds



def Help():
    print("\nThis script calculates 4 sesonal composite for the indicated year, path and row of Landsat images."
          "Required parameters :\n"\
          "  -i,     the path to the directory which contains all the tif images \n"\
          "  -o,     the path where the seasonal composite will be saved \n"\
          "  -c,     the path to the folder, which contains cloud masks for images to be processed \n"\
          "  -s,     the path to the folder, which contains shadow masks for images to be processed"\
          "  -y,     the year for which four composites will be created \n"\
          "  -t      the tile of the Sentinel-2 image \n"\
          "  -a      the area to be processed (Africa, Amazonia or Siberia))\n"\
          "  -n      compute NDVI, assaign 1 if you want to run the script to only compute NDVI "\
          "  -u      number of processors for multiprocessing "\
          "  -d      the sensor type (Sentinel or Landsat) \n"
          
          )
    sys.exit(2)



if __name__ == "__main__":   
    
    tile = 0 
    ndvi = 0


    try:
       opts, args = getopt.getopt(sys.argv[1:],"i:o:c:s:y:d:t:a:n:u:",["help"])
    except getopt.GetoptError:
       print('Invalid argument!')
       sys.exit(2)
    for opt, arg in opts:
       if opt in ("--help"):
          Help()
       elif opt in ("-i"):
          input_path = arg
       elif opt in ("-o"):
          output_path = arg
       elif opt in ("-c"):
          cloudPath = arg
       elif opt in ("-s"):
          shadowPath = arg
       elif opt in ("-y"):
           year = arg
       elif opt in ("-d"):
          sensor = arg
       elif opt in ("-t"):
           tile = arg
       elif opt in ("-a"):
           area = arg
       elif opt in ("-n"):
           ndvi = arg
       elif opt in ("-u"):
          nbOfProcesses = arg
          nbOfProcesses = int(nbOfProcesses)
          
          
    if (not opts):
       Help()
    #Log and configs init
    config_file = "configuration_processor.json"
    with open(config_file) as json_file:
        config = json.load(json_file)

    #--------------------------------------------------------------------------
    #--- LOAD CONFIGURATION
    #--------------------------------------------------------------------------
    verbose = config["verbose"]
    log_level = config["log_level"]

    #--------------------------------------------------------------------------
    #--- LOG SETUP
    #--------------------------------------------------------------------------

    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)
    # log_filename.mkdir(parents=True, exist_ok=True)
    output_path =pathlib.Path("/output")
    log_filename = output_path / 'log_{}.log'.format(dt.now().strftime("%Y-%m-%d-%H-%M"))


    # Add stdout handler, with level INFO
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.getLevelName(log_level.upper()))
    formatter_1 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
    console.setFormatter(formatter_1)
    logging.getLogger().addHandler(console)

    # Add file rotating handler, with level DEBUG
    fileHandler = logging.FileHandler(filename=log_filename)
    fileHandler.setLevel(logging.getLevelName(log_level.upper()))
    formatter_2 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
    fileHandler.setFormatter(formatter_2)
    logging.getLogger().addHandler(fileHandler)

    log = logging.getLogger(__name__)
    log.info("Configuration Complete.")

    #--------------------------
    #----- END LOGSETUP
    #--------------------------
    log.info("Start processing")   

    if ndvi == 1:
         if area == "Siberia" or area == "siberia":
             computeNDVISiberia(input_path, cloudPath, shadowPath, output_path)
         else:             
             computeNDVI(input_path, cloudPath, output_path)
    else:
        if area == "Siberia" or area == "siberia":
            if sensor == "Sentinel":   # Sentinel
               getSiberiaCompositeMedianaS2Parralel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses)      
               #getSiberiaCompositeMedianaS2(input_path, output_path, cloudPath , shadowPath, year)
            else:           # Landsat
               #coregisterImagesSiberia(input_path, cloudPath , shadowPath, area, path, row, year)
               getSesonalCompositeMedianaLandsatSiberia(input_path, output_path,  tile,  year, cloudPath, shadowPath )
               getCumulativeCloudMaskSiberia(input_path, cloudPath, shadowPath, output_path, tile, year)

        else:
            if sensor == "Sentinel":  # Sentinel
               getMontlyCompositeMedianaS2Parralel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses)
               #getMontlyCompositeMedianaS2(input_path, output_path, cloudPath , shadowPath, year)
            else:          # Landsat
               #coregisterImages(input_path, cloudPath , shadowPath, area, path, row, year)
               getSesonalCompositeMedianaLandsat(input_path, output_path,  tile, year, cloudPath, shadowPath )
               getCumulativeCloudMask(input_path, cloudPath, shadowPath, output_path, tile, year)

    log.info("Processing complete")
    

