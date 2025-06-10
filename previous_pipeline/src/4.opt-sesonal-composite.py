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


from osgeo import gdal
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

from utility import *

##############################################################################
# The band order of the output Sentinel-2 composites:
#    'B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12'

# The band order of the output Landsat 5/7 composites:
#    'band1','band2','band3','band4','band5','band7'

# The band order of the output Landsat 8 composites:
#    'band2','band3','band4','band5','band6', 'band7'
##############################################################################

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
  
#####################################################################
#   General - Function with multiple call from different location   #
#####################################################################

def getImagesInBetween(images, after, before, year):
    imagesInBetween = []

    for z in range(len(images)):
            img = os.path.basename(images[z])

            img_date = img[img.index(year):img.index(year)+8]
            img_date = datetime.strptime(img_date, "%Y%m%d")         
            if after < img_date < before:
                imagesInBetween.append(images[z])
    return imagesInBetween

###############################
#   Time series manipulation  #
###############################

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

def loadTimeSeries3DLandsat(landsat_img,pix_region, patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, nb):
    ts_3D = np.zeros((patch_sizeX*patch_sizeY, len(landsat_img), nb))
    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)

            cloud_shadow_mask = (cmask + smask)
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patch_sizeY, patch_sizeX, nb))
            
            for b in range(nb):
                    img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)    
                   # not_cloud_shadow_mask[img_temp[:,:,b] > 55535] = 0  # removes Landsat 7 stripes 
                    img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)
                    
                    
            img_temp[img_temp==0]=['nan']
            img_temp =  np.reshape(img_temp, (patch_sizeX*patch_sizeY, nb))
           
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


def loadTimeSeries3DLandsatSiberia(landsat_img,pix_region, patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, snowMasksList, nb):
    ts_3D = np.zeros((patch_sizeX*patch_sizeY, len(landsat_img), nb))
    ts_3D2 = np.zeros((patch_sizeX*patch_sizeY, len(landsat_img), nb))

    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)
            
            snowMasks_ds = gdal.Open(snowMasksList[i])
            snowMasks = snowMasks_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)

            cloud_shadow_mask = (cmask + smask + snowMasks)
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patch_sizeY, patch_sizeX, nb))
            img_tempNoMask= np.zeros((patch_sizeY, patch_sizeX, nb))

            for b in range(nb):
                    img_tempNoMask[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)    
                   # img_tempNoMask[(img_tempNoMask[:,:,b] > 55535) & (img_tempNoMask[:,:,b] < 55550)] = 0  #strips for L7 and Landsat borders
                   # not_cloud_shadow_mask[img_tempNoMask[:,:,b] > 55535] = 0  # removes Landsat 7 stripes 
                    img_temp[:,:,b] = np.multiply(img_tempNoMask[:,:,b], not_cloud_shadow_mask)
                    
            img_temp[img_temp==0]=['nan']
            img_temp =  np.reshape(img_temp, (patch_sizeX*patch_sizeY, nb))
            img_tempNoMask =  np.reshape(img_tempNoMask, (patch_sizeX*patch_sizeY, nb))
           
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

def loadTimeSeriesMedian3DSiberia(n_imgs, pix_region, patch_size, cloudMasks, shadowMasks, rasters_count):
    
    pid = os.getpid()  # Get the process ID
    ts_3D = np.zeros((patch_size* patch_size, len(n_imgs), rasters_count))
    cumulativeCloudMask = np.zeros(((patch_size, patch_size))) 

    print(f"[P_{pid}] Loading 3D Siberian time serie.")
    for i in range(len(n_imgs)):  

        cmask_ds = gdal.Open(cloudMasks[i])
        cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)          
        smask_ds = gdal.Open(shadowMasks[i])
        smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
        cloud_shadow_mask = (cmask + smask)
        not_cloud_shadow_mask = cloud_shadow_mask^1
        
        img_temp_ds = gdal.Open(n_imgs[i])
        img_temp= np.zeros((patch_size, patch_size, rasters_count))
        
        for b in range(rasters_count):        
            img_temp[:, :, b] = img_temp_ds.GetRasterBand(b + 1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
            img_temp[:,:,b] = np.multiply(img_temp[:,:,b], not_cloud_shadow_mask)

        noData= np.count_nonzero(img_temp, axis=2)
        noData[noData>0] =1
        cumulativeCloudMask = cumulativeCloudMask + noData
        
        img_temp[img_temp==0]=['nan']       
        img_temp =  np.reshape(img_temp, (patch_size*patch_size, rasters_count))      
        ts_3D[:, i, : ] = img_temp
        median = np.nanmedian(ts_3D, axis = 1)
        
    print(f"[P_{pid}] Load complete.")

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

#######################
#   Images alignment  #
#######################

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
    img_temp = target_ds.ReadAsArray(0, 0, int(target_ds.RasterXSize),int(target_ds.RasterYSize))
    img_temp = np.moveaxis(img_temp, 0, -1)

    for k in range(len(landsat_img)-1):
        
        img_temp_ds1 = gdal.Open(landsat_img[k+1])
        img_temp1 = img_temp_ds1.ReadAsArray(0, 0, int(img_temp_ds1.RasterXSize), int(img_temp_ds1.RasterYSize))  
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
        createGeoTif(output_name, aligned[k], 0, target_ds,  gdal.GDT_UInt16)
        
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
    img_temp = target_ds.ReadAsArray(0, 0, int(target_ds.RasterXSize),int(target_ds.RasterYSize))
    
    for k in range(len(landsat_img)-1):
        
        img_temp_ds1 = gdal.Open(landsat_img[k+1])
        img_temp1 = img_temp_ds1.ReadAsArray(0, 0, int(img_temp_ds1.RasterXSize), int(img_temp_ds1.RasterYSize))  
                   
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
        createGeoTifOneBand(output_name, aligned[k], 0, target_ds,  gdal.GDT_Byte)
    #return aligned, target_ds
        
    img_temp = None
    target_ds = None
    img_temp1 = None
    img_temp_ds1 = None
    del target_ds
    del img_temp_ds1

def coregisterImages(input_path, cloudPath , shadowPath, area, path, row, year, configuration):
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
        
        
        

def coregisterImages_Siberia(input_path, cloudPath , shadowPath, area, path, row, year, configuration):
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
    
    
######################################################################
#   NDVI calculation - Functions used with computeOnlyNDVI = True.   #
######################################################################

def computeNDVI(configuration):  
      
    landsat_img_all =  (glob.glob(os.path.join(configuration["tif_images_path"], '*LE07*.tif')))
    cloudMasks_all =  (glob.glob(os.path.join(configuration["cloud_masks_path"], 'LE07*cloudMediumMask.tif')))   
    output_path = configuration["ndvi_output_path"]

    print(f"\nComputing NDVI for {configuration['area']} area in {configuration['year']}:\n")
    if VERBOSE:
        print_list(landsat_img_all, f"Identified {len(landsat_img_all)} images")
        print_list(cloudMasks_all, f"Identified {len(cloudMasks_all)} cloud masks")

    for i in range(len(landsat_img_all)):

        img_ds = gdal.Open(landsat_img_all[i])          
        cloud_ds = gdal.Open(cloudMasks_all[i])            
        cloud = cloud_ds.GetRasterBand(1).ReadAsArray(0, 0, int(cloud_ds.RasterXSize), int(cloud_ds.RasterYSize))
        not_cloud_mask = cloud^1
        band4 = (img_ds.GetRasterBand(4).ReadAsArray(0, 0, int(img_ds.RasterXSize), int(img_ds.RasterYSize))).astype(float)
        band3 = (img_ds.GetRasterBand(3).ReadAsArray(0, 0, int(img_ds.RasterXSize), int(img_ds.RasterYSize))).astype(float)
        
        print(f"\nProcessing image {i+1}/{len(landsat_img_all)}")
        print(f" - Image: {img_ds}")
        print(f" - Cloud mask: {cloud_ds}")
        print(f" - Band 4 shape: {band4.shape} ({type(band4)})")
        print(f" - Band 3 shape: {band3.shape} ({type(band3)})")
        
        ndvi = (band4-band3)/(band4+band3 + 0.000001)
        ndvi = ndvi * not_cloud_mask
        ndvi[ndvi==0] = np.nan  
        ndvi[ndvi>1] = 1  
        ndvi[ndvi<-1] = -1  

        filename = os.path.basename(landsat_img_all[i])
        name = os.path.join(output_path, filename[:-4] + '_NDVI.tif')

        print(f" - Saving NDVI image at: {name}")
        
        createGeoTifOneBand(name, ndvi,  0, img_ds,  gdal.GDT_Float32)
        
    cloud = None
    cloud_ds = None
    del cloud
    del cloud_ds
    
    
def computeNDVISiberia(configuration): 

    landsat_img_all =  (glob.glob(os.path.join(configuration["tif_images_path"], '*LE07*.tif')))
    cloudMasks_all =  (glob.glob(os.path.join(configuration["cloud_masks_path"], 'LE07*cloudMediumMask.tif')))   
    shadowMasks_all =  (glob.glob(os.path.join(configuration["shadow_masks_path"], 'LE07*shadowMask.tif')))   
    snowMasks_all =  (glob.glob(os.path.join(configuration["snow_masks_path"], 'LE07*snowIceMask.tif')))  
    output_path = configuration["ndvi_output_path"]


    if VERBOSE:
        print_list(landsat_img_all, f"Identified {len(landsat_img_all)} images")
        print_list(cloudMasks_all, f"Identified {len(cloudMasks_all)} cloud masks")
        print_list(shadowMasks_all, f"Identified {len(shadowMasks_all)} shadow masks")
        print_list(snowMasks_all, f"Identified {len(snowMasks_all)} snow masks")

    year = str(configuration["year"])
    months = []   
    months.append(['0401', '0630'])
    months.append(['0701', '0930'])
    restoreIx = 2
    
    for s in range(len(months)):

        month = months[s] 
        after = datetime.strptime(year+month[0], "%Y%m%d")
        before = datetime.strptime(year+month[1], "%Y%m%d")

        print(f"\n\nProcessing month {s+1}/12: {month} (B/A: {before} - {after})")

        landsat_img = sorted(getImagesInBetween(landsat_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))
        snowMasks = sorted(getImagesInBetween(snowMasks_all, after, before, year))
        if VERBOSE:
             print_list(landsat_img, f"Founded {len(landsat_img)} Images for current month:")
             print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
             print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
        

        if len(landsat_img) == 0:
            restoreIx = s

        else:

            #first create composite
            target_ds = gdal.Open(landsat_img[0])
            if VERBOSE:
                 gdal_file_info(target_ds)

            raster_x=int(target_ds.RasterXSize)
            raster_y =int(target_ds.RasterYSize)

            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))
                            
            patch_sizeY = math.floor(raster_y / 5)   
            patch_sizeX = math.floor(raster_x / 5)      
            patch_sizeXorig = math.floor(raster_x / 5)
            patch_sizeYorig = math.floor(raster_y / 5)   
            total = (raster_x // patch_sizeX) * (raster_y // patch_sizeY)

            if VERBOSE:
                print(f" - Images: {len(landsat_img)}")
                print(f" - Patch size: ({patch_sizeX},{patch_sizeY})")
                print(f" - Rasters count: {raster_count}")
                print(f" - Parallel processors: {configuration['multiprocessing_precessors']}")

            if (raster_x % patch_sizeX != 0) or (raster_y % patch_sizeY != 0):
                raise IOError(f"Patch dimensions are not correct.\ntarget shape: ({raster_x},{raster_y},{raster_count}),\nPatch size: ({patch_sizeX},{patch_sizeY})")

            print(f"Processing {total} patches:")
            for i in range(5):
                for j in range(5):

                        if i == 4 and patch_sizeX * 5 != raster_x:
                            patch_sizeX = (raster_x - patch_sizeXorig * 5) + patch_sizeXorig - 1
                        else: 
                            patch_sizeX = math.floor(raster_x / 5)
        
                        if j == 4 and patch_sizeY * 5 != raster_y:
                            patch_sizeY = (raster_y - patch_sizeYorig * 5) + patch_sizeYorig - 1
                        else: 
                            patch_sizeY = math.floor(raster_y / 5)
                            
                        pix_region = np.array([i * patch_sizeXorig, j * patch_sizeYorig])

                        print(f"Loading Landsat 3D time series for Siberia with Mediana method for patch {(i*j) + 1}/{total}")
                        img3d, img3d2 = loadTimeSeries3DLandsatSiberia(landsat_img, pix_region,  patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, snowMasks, raster_count)

                        temp = np.nanmedian(img3d, axis = 1)
                        is_all_zero = np.all(np.isnan(temp), axis = 1)
                        #in case there is no data fill with minimum
                        minTable = np.min(img3d2, axis = 1)
                        temp[is_all_zero,:] = minTable[is_all_zero,:]

                        sesonalComposite[j*patch_sizeYorig:j*patch_sizeYorig+patch_sizeY, i*patch_sizeXorig:i*patch_sizeXorig+patch_sizeX, :] =  np.reshape(temp ,[patch_sizeY,patch_sizeX, raster_count])
                    
            #compute NDVi
            band4 = sesonalComposite[:,:,4] 
            band3 = sesonalComposite[:,:,3]
            ndvi = (band4-band3)/(band4+band3 + 0.000001)
            ndvi[ndvi==0] = np.nan  
            ndvi[ndvi>1] = 1  
            ndvi[ndvi<-1] = -1  
            
            name = os.path.join(output_path,  'NDVI_'+ str(s) +'.tif')
            print(f"Saving generated NDVI as: {name}")
            createGeoTifOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
    
    #if for one composite no images are avaiable
    if restoreIx  == 0:
        otherNDVIname = os.path.join(output_path,  'NDVI_1.tif')
        target_ds = gdal.Open(otherNDVIname)            
        ndvi = target_ds.GetRasterBand(1).ReadAsArray(0, 0, target_ds.RasterXSize, target_ds.RasterYSize) 
        
        name = os.path.join(output_path,  'NDVI_ '+ str(restoreIx) +'.tif')
        createGeoTifOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
    
    elif restoreIx  == 1:
        otherNDVIname = os.path.join(output_path,  'NDVI_0.tif')
        target_ds = gdal.Open(otherNDVIname)
        ndvi = target_ds.GetRasterBand(1).ReadAsArray(0, 0, target_ds.RasterXSize, target_ds.RasterYSize) 
        name = os.path.join(output_path,  'NDVI_ '+ str(restoreIx) +'.tif')
        createGeoTifOneBand(name, ndvi,  0, target_ds,  gdal.GDT_Float32)
        
    #DON'T forget  about NDVI restoration!!!!
    
    cloud = None
    cloud_ds = None
    target_ds = None

    del cloud
    del cloud_ds
    del target_ds

########################################################################################
#   Sentinel Siberia composites with Mediana method - Not parallelize & parallelized   #
########################################################################################

def getSiberiaComposite_Sentinel_Mediana(input_path, output_path, cloudPath , shadowPath, year, configuration):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    sentinel_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))
    if VERBOSE:
        print_list(sentinel_img_all, f"Founded {len(sentinel_img_all)} Images in total:")
        print_list(cloudMasks_all, f"Founded {len(cloudMasks_all)} Cloud masks in total:")
        print_list(shadowMasks_all, f"Founded {len(shadowMasks_all)} Shadows masks in total:")
        
    months = []
    months.append(['07', '08'])
    year = str(configuration["year"])

    for s in range(len(months)):   

        # Identify data for the current month
        month = months[s]
        print(f"\n\nProcessing month {s}/{len(month)}: {month}\n")

        cloudMasks = sorted([x for x in cloudMasks_all if year+month[0] in x or year+month[1] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year+month[0] in x or year+month[1] in x ])
        sentinel_imgs = sorted([x for x in sentinel_img_all if year+month[0] in x or year+month[1] in x ])
        if VERBOSE:
             print_list(sentinel_imgs, f"Founded {len(sentinel_imgs)} Images for current month:")
             print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
             print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
        
        # Data validity check
        if len(cloudMasks) != len(sentinel_imgs) or len(shadowMasks) != len(sentinel_imgs):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")

        if len(cloudMasks) != 0:

            # Read .tif files & print info
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(sentinel_imgs[0])
            if VERBOSE:
                 gdal_file_info(cmask_ds)
                 gdal_file_info(target_ds)

            # Initialize composite

            raster_y = int(target_ds.RasterYSize)
            raster_x = int(target_ds.RasterXSize)
            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))            
            cumulativeCloudMask = np.zeros((raster_x, raster_y))

            #for i in range(1):
            patch_sizeX = 2196 
            patch_sizeY = 2196    
            total = (raster_x // patch_sizeX) * (raster_y // patch_sizeY)
            
            if VERBOSE:
                print(f" - Images: {len(sentinel_imgs)}")
                print(f" - Patch size: ({patch_sizeX},{patch_sizeY})")
                print(f" - Rasters count: {raster_count}")
                print(f" - Parallel processors: {configuration['multiprocessing_precessors']}")

            if (raster_x % patch_sizeX != 0) or (raster_y % patch_sizeY != 0):
                raise IOError(f"Patch dimensions are not correct.\ntarget shape: ({raster_x},{raster_y},{raster_count}),\nPatch size: ({patch_sizeX},{patch_sizeY})")

            # Process each patch and build composite and cumulative cloud mask

            print(f"Processing {total} patches:")
            for i in range(raster_x// patch_sizeX):
                for j in range(raster_y // patch_sizeY):
                    
                    pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                    
                    print(f"\nLoading Sentinel 3D time series for Siberia with Mediana method for patch {(i*j) + 1}/{total}")
                    img3d, cumulativeCloudMaskTemp = loadTimeSeriesMedian3DSiberia(sentinel_imgs, pix_region, patch_sizeX, cloudMasks, shadowMasks, raster_count)
                    #temp = np.nanmedian(img3d, axis = 1)
                    
                    sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(img3d ,[patch_sizeX,patch_sizeY, raster_count])
                    cumulativeCloudMask[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  cumulativeCloudMaskTemp

            # Save composite and cumulative cloud mask
            composite_path = os.path.join(str(output_path),'yearlyCompositeMedianMasked_'+ str(tile) + str(year) + '_' + str(s+1) + '.tif')
            print(f"Saving generated sasonal composite as: {composite_path}")
            createGeoTif(composite_path, sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            
            cloud_cum_mask_path = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + '_' + str(s+1) + '.tif')      
            print(f"Saving generated cumulative cloud mask as: {cloud_cum_mask_path}")
            createGeoTifOneBand(cloud_cum_mask_path, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)     
 
             # Free resources
            target_ds = None
            del target_ds
            cmask_ds = None
            del cmask_ds

def getSiberiaComposite_Sentinel_Mediana_Parallel(input_path, output_path, cloudPath , shadowPath, year, configuration):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*cloudMediumMask.tif')))   
    shadowMasks_all = sorted(glob.glob(os.path.join(shadowPath, '*shadowMask.tif')))   
    sentinel_img_all =  sorted(glob.glob(os.path.join(input_path, '*.tif')))
    if VERBOSE:
        print_list(sentinel_img_all, f"Founded {len(sentinel_img_all)} Images in total:")
        print_list(cloudMasks_all, f"Founded {len(cloudMasks_all)} Cloud masks in total:")
        print_list(shadowMasks_all, f"Founded {len(shadowMasks_all)} Shadows masks in total:")
        
    months = []
    months.append(['07', '08'])
    year = str(configuration["year"])
    cumulativeCloudMask = np.zeros(((10980, 10980)))

    for s in range(len(months)):   

        # Identify data for the current month
        month = months[s]
        print(f"\n\nProcessing month {s}/{len(month)}: {month}\n")

        cloudMasks = sorted([x for x in cloudMasks_all if year+month[0] in x or year+month[1] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year+month[0] in x or year+month[1] in x ])
        sentinel_imgs = sorted([x for x in sentinel_img_all if year+month[0] in x or year+month[1] in x ])
        if VERBOSE:
             print_list(sentinel_imgs, f"Founded {len(sentinel_imgs)} Images for current month:")
             print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
             print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
        
        # Data validity check
        if len(cloudMasks) != len(sentinel_imgs) or len(shadowMasks) != len(sentinel_imgs):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")

        if len(cloudMasks) != 0:
            
            # Read .tif files & print info
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(sentinel_imgs[0])
            if VERBOSE:
                gdal_file_info(cmask_ds)
                gdal_file_info(target_ds)

            # Initialize composite
            raster_y = int(target_ds.RasterYSize)
            raster_x = int(target_ds.RasterXSize)
            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))

            values=[]
            patch_sizeX = 1098  #2196
            patch_sizeY = 1098  #2196        
            total =(raster_x // patch_sizeX) * (raster_y // patch_sizeY)

            if VERBOSE:
                print(f" - Images: {len(sentinel_imgs)}")
                print(f" - Patch size: ({patch_sizeX},{patch_sizeY})")
                print(f" - Rasters count: {raster_count}")
                print(f" - Parallel processors: {configuration['multiprocessing_precessors']}")

            # Check patch shape in relation to the target shape
            if (raster_x % patch_sizeX != 0) or (raster_y % patch_sizeY != 0):
                raise IOError(f"Patch dimensions are not correct.\ntarget shape: ({raster_x},{raster_y},{raster_count}),\nPatch size: ({patch_sizeX},{patch_sizeY})")

            # Process each patch parameters for parallel processing.
            # Identify the tuple of parameters for function loadTimeSeriesMedian3DSiberia
            print(f"Processing {total} patches ... ", end = "", flush=True)
            for i in range(raster_x // patch_sizeX):
                for j in range(raster_y // patch_sizeY):
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        values.append((sentinel_imgs, pix_region, patch_sizeX, cloudMasks, shadowMasks, raster_count))              
            
            values = tuple(values)
            print("Parameters retrieved.")
            
            # Parallel region - Will spawn a process for each patch (total)
            print("Loading Sentinel 3D time series for Siberia with Mediana method in parallel ... ")
            p = Pool(configuration["multiprocessing_precessors"])
            result = p.starmap(loadTimeSeriesMedian3DSiberia, values)
            # p.close()
            # p.join()

            # Build final composite
            k = 0
            for i in range(10):
                for j in range(10):
                    temp = result[k]
                    sesonalComposite[j * patch_sizeX : j * (patch_sizeX + patch_sizeX), i * patch_sizeX : i * (patch_sizeX + patch_sizeX) ] =  np.reshape(temp[0], [patch_sizeX, patch_sizeY, raster_count])
                    cumulativeCloudMask[j * patch_sizeX : j * (patch_sizeX + patch_sizeX), i * patch_sizeX : i * (patch_sizeX + patch_sizeX) ] =  temp[1]
                    k += 1           
           
            # Save the composite and the cloud cumulative mask
            composite_path = os.path.join(str(output_path),'yearlyCompositeMedianMaskedParaler_' + tile + year + '_' + str(s+1) + '.tif')
            print(f"Saving generated sasonal composite as: {composite_path}")
            createGeoTif(composite_path, sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            
            cloud_cum_mask_path = 'cumulativeCloudMask_' + str(tile) + str(year) + '_'  + '.tif'        
            print(f"Saving generated cumulative cloud mask as: {cloud_cum_mask_path}")
            createGeoTifOneBand(cloud_cum_mask_path, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
            
            # Free resources
            target_ds = None
            del target_ds
            cmask_ds = None
            del cmask_ds




##############################################################################
#   Sentinel monthly composites with Mediana method - Not parallelize & parallelized   #
##############################################################################

def getMontlyComposite_Sentinel_Mediana(input_path, output_path, cloudPath , shadowPath, year, configuration ):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*MSIL2A*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*MSIL2A*shadowMask.tif')))   
    sentinel_img_all =  sorted(glob.glob(os.path.join(input_path, '*MSIL2A*.tif')))   
    if VERBOSE:
        print_list(sentinel_img_all, f"Founded {len(sentinel_img_all)} Images in total:")
        print_list(cloudMasks_all, f"Founded {len(cloudMasks_all)} Cloud masks in total:")
        print_list(shadowMasks_all, f"Founded {len(shadowMasks_all)} Shadows masks in total:")
        
    year = str(configuration["year"])

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
 
         # Identify data for the current month
        month = months[s] 
        after = datetime.strptime(year + month[0], "%Y%m%d")
        before = datetime.strptime(year + month[1], "%Y%m%d")

        print(f"\n\nProcessing month {s+1}/12: {month} (B/A: {before} - {after})")

        sentinel_imgs = sorted(getImagesInBetween(sentinel_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))
        if VERBOSE:
             print_list(sentinel_imgs, f"Founded {len(sentinel_imgs)} Images for current month:")
             print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
             print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
        
        # Data validity check
        if len(cloudMasks) != len(sentinel_imgs) or len(shadowMasks) != len(sentinel_imgs):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")
  
        if len(cloudMasks) != 0:

            # Read .tif files & print info
            target_ds = gdal.Open(sentinel_imgs[0])
            if VERBOSE:
                 gdal_file_info(target_ds)

            # Initialize composite
            raster_y = int(target_ds.RasterYSize)
            raster_x = int(target_ds.RasterXSize)
            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))

            #for i in range(1): 
            patch_sizeX = 2196 
            patch_sizeY = 2196                
            total = (raster_x // patch_sizeX) * (raster_y // patch_sizeY)

            if VERBOSE:
                print(f" - Images: {len(sentinel_imgs)}")
                print(f" - Patch size: ({patch_sizeX},{patch_sizeY})")
                print(f" - Rasters count: {raster_count}")
                print(f" - Parallel processors: {configuration['multiprocessing_precessors']}")

            if (raster_x % patch_sizeX != 0) or (raster_y % patch_sizeY != 0):
                raise IOError(f"Patch dimensions are not correct.\ntarget shape: ({raster_x},{raster_y},{raster_count}),\nPatch size: ({patch_sizeX},{patch_sizeY})")

            print(f"Processing {total} patches:")
            for i in range(raster_x // patch_sizeX): 
                for j in range(raster_y // patch_sizeY): 

                    pix_region = np.array([i*patch_sizeX, j*patch_sizeX])

                    print(f"Loading Sentinel 3D time series for patch {(i*j) + 1}/{total}")
                    img3d = loadTimeSeries3D(sentinel_imgs, pix_region, patch_sizeX, cloudMasks, shadowMasks, raster_count)
                    temp = np.nanmedian(img3d, axis = 1)
                    #tempMin = np.nanmin(img3d, axis=1)
                    
                    sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp ,[patch_sizeX,patch_sizeY, raster_count])
                    
            # Save composite & free resources
            composite_path = os.path.join(str(output_path),'montlyCompositeMedianMasked_' + tile + year + '_' + str(s+1) + '.tif')
            print(f"Saving generated sasonal composite as: {composite_path}")
            createGeoTif(composite_path, sesonalComposite, 0, target_ds, gdal.GDT_UInt16)
            
            target_ds = None
            del target_ds

def getMonthlyComposite_Sentinel_Mediana_Parallel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses, configuration):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks_all =  sorted(glob.glob(os.path.join(cloudPath, '*MSIL2A*cloudMediumMask.tif')))   
    shadowMasks_all =  sorted(glob.glob(os.path.join(shadowPath, '*MSIL2A*shadowMask.tif')))   
    sentinel_img_all =  sorted(glob.glob(os.path.join(input_path, '*MSIL2A*.tif')))   
    if VERBOSE:
        print_list(sentinel_img_all, f"Founded {len(sentinel_img_all)} Images in total:")
        print_list(cloudMasks_all, f"Founded {len(cloudMasks_all)} Cloud masks in total:")
        print_list(shadowMasks_all, f"Founded {len(shadowMasks_all)} Shadows masks in total:")
        
    year = str(configuration["year"])

    if len(cloudMasks_all) != len(sentinel_img_all) or len(shadowMasks_all) != len(sentinel_img_all):
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

        # Identify data for the current month
        month = months[s] 
        after = datetime.strptime(year+month[0], "%Y%m%d")
        before = datetime.strptime(year+month[1], "%Y%m%d")
        
        print(f"\n\nProcessing month {s+1}/12: {month} (B/A: {before} - {after})")

        sentinel_imgs = sorted(getImagesInBetween(sentinel_img_all, after, before, year))
        cloudMasks = sorted(getImagesInBetween(cloudMasks_all, after, before, year))
        shadowMasks = sorted(getImagesInBetween(shadowMasks_all, after, before, year))
        if VERBOSE:
             print_list(sentinel_imgs, f"Founded {len(sentinel_imgs)} Images for current month:")
             print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
             print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
        
        # Data validity check
        if len(cloudMasks) != len(sentinel_imgs) or len(shadowMasks) != len(sentinel_imgs):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")

        if len(cloudMasks) != 0:

            # Read .tif files & print info
            cmask_ds = gdal.Open(cloudMasks[0])
            target_ds = gdal.Open(sentinel_imgs[0])
            if VERBOSE:
                gdal_file_info(cmask_ds)
                gdal_file_info(target_ds)

            raster_y = int(target_ds.RasterYSize)
            raster_x = int(target_ds.RasterXSize)
            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))
           
            values=[]
            patch_sizeX = 1098  #1098 2196
            patch_sizeY = 1098  #1098  2196     
            total =(raster_x // patch_sizeX) * (raster_y // patch_sizeY)

            if VERBOSE:
                print(f" - Images: {len(sentinel_imgs)}")
                print(f" - Patch size: ({patch_sizeX},{patch_sizeY})")
                print(f" - Rasters count: {raster_count}")
                print(f" - Parallel processors: {configuration['multiprocessing_precessors']}")

            # Check patch shape in relation to the target shape
            if (raster_x % patch_sizeX != 0) or (raster_y % patch_sizeY != 0):
                raise IOError(f"Patch dimensions are not correct.\ntarget shape: ({raster_x},{raster_y},{raster_count}),\nPatch size: ({patch_sizeX},{patch_sizeY})")

            # Process each patch parameters for parallel processing.
            # Identify the tuple of parameters for function loadTimeSeriesMedian3DSiberia
            print(f"Processing {total} patches ... ", end = "", flush=True)
            for i in range(raster_x // patch_sizeX):   
                for j in range(raster_y // patch_sizeY):
                        pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                        values.append((sentinel_imgs, pix_region, patch_sizeX, cloudMasks, shadowMasks, raster_count))              
            values = tuple(values)
            print("Parameters retrieved.")

            # Parallel region - Will spawn a process for each patch (total)
            print("Loading Sentinel 3D time series with Mediana Cumulative method in parallel ... ")
            p = Pool(nbOfProcesses)
            result = p.starmap(loadTimeSeriesMedian3DCumulative, values)
            # p.close()
            ## p.join()

            k=0
            for i in range(10): 
                for j in range(10):            
                    temp = result[k]
                    sesonalComposite[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp[0], [patch_sizeX,patch_sizeY, raster_count])
                    cumulativeCloudMask[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  temp[1]
                    k = k+1                         

            # Save the composite and the cloud cumulative mask
            composite_path = os.path.join(str(output_path),'montlyCompositeMedianMasked_' + str(tile) + str(year) + '_'  + str(s+1) + '.tif')
            print(f"Saving generated sasonal composite as: {composite_path}")
            createGeoTif(composite_path, sesonalComposite, 0, target_ds, gdal.GDT_UInt16)

            cloud_cum_mask_path = os.path.join(str(output_path),'montlycumulativeCloudMask_' + str(tile) + str(year) + '_'  + str(s+1) + '.tif' )       
            print(f"Saving generated cumulative cloud mask as: {cloud_cum_mask_path}")
            createGeoTifOneBand(cloud_cum_mask_path, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
   
            # Free resources
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
                cumulativeCloudMaskTemp = cumulativeCloudMaskTemp_ds.ReadAsArray(0, 0, int(nc),int(nl))
                cumulativeCloudMaskUnited  = cumulativeCloudMaskUnited + cumulativeCloudMaskTemp
                            
                months_used = months_used + '_' + re.search('%s(.*)%s' % (start, end),cumulativeMasks_all[p]).group(1)
                
            outputName = os.path.join(str(output_path),'montlycumulativeCloudMask_' + str(tile) + str(year) + str(months_used) + '.tif' )       
            createGeoTifOneBand(outputName, cumulativeCloudMaskUnited, 0, cumulativeCloudMaskTemp_ds,  gdal.GDT_Byte)   
            cumulativeCloudMaskTemp_ds = None
            del cumulativeCloudMaskTemp_ds


###############################################################################
#   Landsat seasonal composite with Mediana method - Standard & For Siberia   #
###############################################################################

def getSeasonalComposite_Landsat_Mediana(input_path, output_path, tile, year, cloudPath , shadowPath, configuration ):
    
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
            xsize=int(target_ds.RasterXSize) 
            ysize =int(target_ds.RasterYSize)
        
            nb = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((ysize, xsize, nb)))
            
            for i in range(len(landsat_img)):
                patch_sizeY = math.floor(ysize/5)   
                patch_sizeX = math.floor(xsize/5)      
                patch_sizeXorig = math.floor(xsize/5)
                patch_sizeYorig = math.floor(ysize/5)   

                for i in range(5):
                    for j in range(5):

                       if i==4 and patch_sizeX*5 != xsize:
                           patch_sizeX = (xsize - patch_sizeXorig*5) + patch_sizeXorig-1
                       else: 
                           patch_sizeX = math.floor(xsize/5)
        
                       if j==4 and patch_sizeY*5 != ysize:
                           patch_sizeY = (ysize - patch_sizeYorig*5) + patch_sizeYorig-1
                       else: 
                           patch_sizeY = math.floor(ysize/5)
                           
                       pix_region = np.array([i*patch_sizeXorig, j*patch_sizeYorig])
                       img3d = loadTimeSeries3DLandsat(landsat_img, pix_region,  patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, nb)

                       temp = np.nanmedian(img3d, axis = 1)
                       sesonalComposite[j*patch_sizeYorig:j*patch_sizeYorig+patch_sizeY, i*patch_sizeXorig:i*patch_sizeXorig+patch_sizeX, :] =  np.reshape(temp ,[patch_sizeY,patch_sizeX, nb])

  
            createGeoTif(os.path.join(str(output_path),'sesonalCompositeMedianMasked_' + str(tile) + year +'_' + str(s+1) + '.tif'), sesonalComposite, 0, target_ds, gdal.GDT_UInt16)   

            target_ds = None
            del target_ds            

            
   
def getSiberiaSeasonalComposite_Landsat_Mediana(input_path, output_path,  tile,  year, cloudPath, shadowPath, configuration ):
    
    np.seterr(divide='ignore', invalid='ignore')
    landsat_img = sorted(glob.glob(os.path.join(input_path, 'LE07*.tif')))
    cloudMasks =   sorted(glob.glob(os.path.join(cloudPath,  'LE07*cloudMediumMask.tif')))    
    shadowMasks =  sorted(glob.glob(os.path.join(shadowPath, 'LE07*shadowMask.tif')))   
    snowMasks = sorted(glob.glob(os.path.join(configuration["snow_masks_path"],  'LE07*snowIceMask.tif')))   

    if VERBOSE:
            print_list(landsat_img, f"Founded {len(landsat_img)} Images for current month:")
            print_list(cloudMasks, f"Founded {len(cloudMasks)} Cloud masks for current month:")
            print_list(shadowMasks, f"Founded {len(shadowMasks)} Shadows masks for current month:")
            print_list(snowMasks, f"Founded {len(snowMasks)} Snow masks for current month:")

    #months = []
    #months.append(['04', '05', '06', '07', '08', '09']) # jan feb march

    for s in range(1): 
                
        # Data validity check
        if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

        if len(cloudMasks) != 0:  
            
            # Read .tif files & print info
            target_ds = gdal.Open(landsat_img[0])
            if VERBOSE:
                gdal_file_info(target_ds)

            raster_x=int(target_ds.RasterXSize)
            raster_y =int(target_ds.RasterYSize)
            raster_count = target_ds.RasterCount                  
            sesonalComposite = np.zeros(((raster_x, raster_y, raster_count)))
            

            # Which i is used inside ? The inner loop i or the outer loop i  ?
            for i in range(1):  #(len(landsat_img)): ??????????????????????
                patch_sizeY = math.floor(raster_y / 5)   
                patch_sizeX = math.floor(raster_x / 5)      
                patch_sizeXorig = math.floor(raster_x / 5)
                patch_sizeYorig = math.floor(raster_y / 5)   
                total = 5 * 5

                for i in range(5):
                    for j in range(5):
                        
                        if i == 4 and patch_sizeX*5 != raster_x:
                            patch_sizeX = (raster_x - patch_sizeXorig * 5) + patch_sizeXorig - 1
                        else: 
                            patch_sizeX = math.floor(raster_x / 5)
            
                        if j == 4 and patch_sizeY * 5 != raster_y:
                            patch_sizeY = (raster_y - patch_sizeYorig * 5) + patch_sizeYorig - 1
                        else: 
                            patch_sizeY = math.floor(raster_y / 5)
                            
                        pix_region = np.array([i * patch_sizeXorig, j * patch_sizeYorig])
                        print(f"\nLoading Landsat 3D time series for Siberia with Mediana method for patch {(i*j) + 1}/{total}")

                        img3d, img3d2 = loadTimeSeries3DLandsatSiberia(landsat_img, pix_region,  patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, snowMasks, raster_count)

                        temp = np.nanmedian(img3d, axis = 1)
                        is_all_zero = np.all(np.isnan(temp), axis = 1)
                        #in case there is no data fill with minimum
                        minTable = np.min(img3d2, axis = 1)
                        temp[is_all_zero,:] = minTable[is_all_zero,:]
                        #border = np.all(temp> 55535, axis = 1)
                        #temp[border,:] = 0

                        sesonalComposite[j*patch_sizeYorig:j*patch_sizeYorig+patch_sizeY, i*patch_sizeXorig:i*patch_sizeXorig+patch_sizeX, :] =  np.reshape(temp ,[patch_sizeY,patch_sizeX, raster_count])
            
            # Save composite
            composite_path = os.path.join(str(output_path),'yearlyCompositeMedianMasked_' + str(tile) + year + '.tif')
            print(f"Saving generated sasonal composite as: {composite_path}")
            createGeoTif(composite_path, sesonalComposite, 0, target_ds, gdal.GDT_UInt16)

            # Free resources
            target_ds = None
            del target_ds

###############################
#   Cloud mask manipulation   #
###############################

def loadCumulativeCloudMask(landsat_img, pix_region, patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, snowMasks=""):
    cumulativeCloudMask = np.zeros((patch_sizeX,patch_sizeY))

    for i in range(len(landsat_img)):    


            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)

            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)
            
            if len(snowMasks) != 0:
                snowMasks_ds = gdal.Open(shadowMasks[i])
                snowMasks = snowMasks_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)
                cloud_shadow_mask = cmask + smask  + snowMasks
            else:
                cloud_shadow_mask = (cmask + smask ) 
                
            cloud_shadow_mask = ndimage.binary_dilation(cloud_shadow_mask)  
            not_cloud_shadow_mask = cloud_shadow_mask^1
           
            img_temp_ds = gdal.Open(landsat_img[i])
            img_temp= np.zeros((patch_sizeY, patch_sizeX))
            img_tempNoMask= np.zeros((patch_sizeY, patch_sizeX))

            for b in range(1):
                    img_tempNoMask[:, :] = img_temp_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_sizeX, patch_sizeY)    
                    img_temp[:,:] = np.multiply(img_tempNoMask[:,:], not_cloud_shadow_mask)
                                        
            img_temp[img_temp!=0]=1
            img_temp =  np.reshape(img_temp, (patch_sizeX,patch_sizeY))
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

def getCumulativeCloudMask(input_path, cloudPath, shadowPath, output_path, tile, year, configuration): 
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
    nl = int(cmask_ds.RasterXSize) 
    nc = int(cmask_ds.RasterYSize)
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
            xsize=int(target_ds.RasterXSize)
            ysize =int(target_ds.RasterYSize)
            
            nb = target_ds.RasterCount                  
            cumulativeCloudMask = np.zeros((ysize, xsize))
                                       
            patch_sizeY = math.floor(ysize/5)   
            patch_sizeX = math.floor(xsize/5)      
            patch_sizeXorig = math.floor(xsize/5)
            patch_sizeYorig = math.floor(ysize/5)   
            
            for i in range(5):
                for j in range(5):
            
                    if i==4 and patch_sizeX*5 != xsize:
                        patch_sizeX = (xsize - patch_sizeXorig*5) + patch_sizeXorig-1
                    else: 
                        patch_sizeX = math.floor(xsize/5)
                    
                    if j==4 and patch_sizeY*5 != ysize:
                          patch_sizeY = (ysize - patch_sizeYorig*5) + patch_sizeYorig-1
                    else: 
                          patch_sizeY = math.floor(ysize/5)
                                       
                    pix_region = np.array([i*patch_sizeXorig, j*patch_sizeYorig])
                    cumulativeCloudMask[j*patch_sizeYorig:j*patch_sizeYorig+patch_sizeY, i*patch_sizeXorig:i*patch_sizeXorig+patch_sizeX] = loadCumulativeCloudMask(landsat_img, pix_region,  patch_sizeX, patch_sizeY, cloudMasks, shadowMasks)
           
            name =  os.path.join(output_path,  'cumulativeCloudMask_'+ (tile) + str(year) + '_'+str(s+1)+'.tif')
            createGeoTifOneBand(name, cumulativeCloudMask,  0, target_ds,  gdal.GDT_Float32)
            
        if s == 3:  # after december has been generated merge all masks together
            start = str(tile) + str(year) + '_'
            end = '.tif'
            cumulativeCloudMaskUnited = np.zeros(((nc, nl)))
            cumulativeMasks_all =  sorted(glob.glob(os.path.join(str(output_path),'cumulativeCloudMask_*.tif')))   
            months_used = ''
            for p in range(len(cumulativeMasks_all)):
                cumulativeCloudMaskTemp_ds = gdal.Open(cumulativeMasks_all[p])
                cumulativeCloudMaskTemp = cumulativeCloudMaskTemp_ds.ReadAsArray(0, 0, int(nl),int(nc))
                cumulativeCloudMaskUnited  = cumulativeCloudMaskUnited + cumulativeCloudMaskTemp
                            
                months_used = months_used + '_' + re.search('%s(.*)%s' % (start, end),cumulativeMasks_all[p]).group(1)
                
            outputName = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + str(months_used) + '.tif' )       
            createGeoTifOneBand(outputName, cumulativeCloudMaskUnited, 0, cumulativeCloudMaskTemp_ds,  gdal.GDT_Byte)   
            cumulativeCloudMaskTemp_ds = None
            del cumulativeCloudMaskTemp_ds
            
    cmask_ds = None
    del cmask_ds
    
def getCumulativeCloudMask_Siberia(input_path, cloudPath, shadowPath, output_path, tile, year, configuration):
    
    np.seterr(divide='ignore', invalid='ignore')
    cloudMasks =   sorted(glob.glob(os.path.join(cloudPath,  '*cloudMediumMask*.tif')))    
    shadowMasks =  sorted(glob.glob(os.path.join(shadowPath, '*shadowMask*.tif')))   
    snowMasks = sorted(glob.glob(os.path.join(cloudPath,  '*snowIceMask*.tif')))   
    landsat_img = sorted(glob.glob(os.path.join(input_path, '*.tif'))) 

    cmask_ds = gdal.Open(cloudMasks[0])
    nl = int(cmask_ds.RasterXSize) 
    nc = int(cmask_ds.RasterYSize)
    cumulativeCloudMask = np.zeros(((nc, nl)))

                
    if len(cloudMasks) != len(landsat_img) or len(shadowMasks) != len(landsat_img):
        raise IOError("The number of clouds and shadow masks should coresponds to the number of Landsat images!")

    if len(cloudMasks) != 0:            
            target_ds = gdal.Open(landsat_img[0])
            xsize=int(target_ds.RasterXSize)
            ysize =int(target_ds.RasterYSize)
            nb = target_ds.RasterCount                  
            
            patch_sizeY = math.floor(ysize/5)   
            patch_sizeX = math.floor(xsize/5)      
            patch_sizeXorig = math.floor(xsize/5)
            patch_sizeYorig = math.floor(ysize/5)   

            for i in range(5):
                for j in range(5):

                       if i==4 and patch_sizeX*5 != xsize:
                           patch_sizeX = (xsize - patch_sizeXorig*5) + patch_sizeXorig-1
                       else: 
                           patch_sizeX = math.floor(xsize/5)
        
                       if j==4 and patch_sizeY*5 != ysize:
                           patch_sizeY = (ysize - patch_sizeYorig*5) + patch_sizeYorig-1
                       else: 
                           patch_sizeY = math.floor(ysize/5)
                           
                       pix_region = np.array([i*patch_sizeXorig, j*patch_sizeYorig])

                       cumulativeCloudMask[j*patch_sizeYorig:j*patch_sizeYorig+patch_sizeY, i*patch_sizeXorig:i*patch_sizeXorig+patch_sizeX] =  loadCumulativeCloudMask(landsat_img, pix_region,  patch_sizeX, patch_sizeY, cloudMasks, shadowMasks, snowMasks)
           
            
    outputName = os.path.join(str(output_path),'cumulativeCloudMask_' + str(tile) + str(year) + '.tif')
    createGeoTifOneBand(outputName, cumulativeCloudMask, 0, cmask_ds,  gdal.GDT_Byte)   
    cmask_ds = None
    del cmask_ds


if __name__ == "__main__":   

    gdal.PushErrorHandler(gdal_error_handler)
    gdal.AllRegister()
    gdal.UseExceptions()
    
    # Read application parameters at the specified path.
    configuration = retrieve_configuration()

    # Build path accordingly to parameters
    configuration["seasonal_composite_output_path"] = configuration["seasonal_composite_output_path"] + f"/{configuration['year']}_{configuration['area']}"
    configuration["ndvi_output_path"] = configuration["seasonal_composite_output_path"] + "/ndvi"
    configuration["cloud_masks_path"] = configuration["masks_path"] + f"/{configuration['year']}_{configuration['area']}/cloud_masks"
    configuration["shadow_masks_path"] = configuration["masks_path"] + f"/{configuration['year']}_{configuration['area']}/shadow_masks"
    configuration["snow_masks_path"] = configuration["masks_path"] + f"/{configuration['year']}_{configuration['area']}/snow_masks"

    VERBOSE = configuration["verbose"]

    create_folder_if_not_exists(configuration["seasonal_composite_output_path"])
    create_folder_if_not_exists(configuration["ndvi_output_path"])

    print(f"\nSeasonal Composite and NDVI index generation\n")
    print_map(configuration, "Application parameters:")

    # TO BE REMOVED
    sensor = configuration["sensor"]
    input_path = configuration["tif_images_path"]
    output_path = configuration["seasonal_composite_output_path"]
    cloudPath = configuration["cloud_masks_path"]
    shadowPath = configuration["shadow_masks_path"]
    year = configuration["year"]
    nbOfProcesses = configuration["multiprocessing_precessors"]
    tile = configuration["tile"]
    area = configuration["area"]
    isSiberianArea = str.lower(area) == "siberia"

    if configuration["computeOnlyNDVI"] is True:
        
        print(f"\n\nComputing NVDI only - Area: {configuration['area']} - Year: {configuration['year']}\n")
        
        if isSiberianArea:
             computeNDVISiberia(configuration)
        else:             
             computeNDVI(configuration)
    
    elif configuration["computeOnlyNDVI"] is False:

        print(f"\n\nGenerating composites for:")
        print(f" - Area: {configuration['area']}")
        print(f" - Year: {configuration['year']}")
        print(f" - Sensor: {configuration['sensor']}\n")

        if sensor == "S2":
            # All logic examined for S2 with both Siberia and Amazzonia. TO DELETE.

            if isSiberianArea: # Sentinel
               getSiberiaComposite_Sentinel_Mediana_Parallel(input_path, output_path, cloudPath , shadowPath, year, configuration)      
               #getSiberiaComposite_Sentinel_Mediana(input_path, output_path, cloudPath , shadowPath, year, configuration)
            
            else:
               getMonthlyComposite_Sentinel_Mediana_Parallel(input_path, output_path, cloudPath , shadowPath, year, nbOfProcesses, configuration)
               #getMontlyComposite_Sentinel_Mediana(input_path, output_path, cloudPath , shadowPath, year, configuration)

        elif sensor in ["L5","L7","L8"]: # Landsat

            if isSiberianArea:
                #coregisterImages_Siberia(input_path, cloudPath , shadowPath, area, path, row, year, configuration)
                getSiberiaSeasonalComposite_Landsat_Mediana(input_path, output_path,  tile,  year, cloudPath, shadowPath, configuration )
                
                print(f"\nGenerating cumulative cloud mask for {configuration['area']} in {configuration['year']}\n")
                getCumulativeCloudMask_Siberia(input_path, cloudPath, shadowPath, output_path, tile, year, configuration)
            
            else:
                #coregisterImages(input_path, cloudPath , shadowPath, area, path, row, year, configuration)
                getSeasonalComposite_Landsat_Mediana(input_path, output_path,  tile, year, cloudPath, shadowPath, configuration )

                print(f"\nGenerating cumulative cloud mask for {configuration['area']} in {configuration['year']}\n")
                getCumulativeCloudMask(input_path, cloudPath, shadowPath, output_path, tile, year, configuration)

        else:
            raise RuntimeError(f"Sensor specified is not available. Sensors: L5, L7, L8, S2. Specified: {configuration['sensor']}")

    else:
        raise RuntimeError(f"Compute only NDVI parameter value specified is not correct. Allowed values are: True / False. Specified: {configuration['computeOnlyNDVI']}")

    print("Processing complete")
    

    """
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
    #--------------------------"""