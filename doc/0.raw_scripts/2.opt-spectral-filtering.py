# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:43:31 2021

@author: rslab
"""

from osgeo import gdal
#import gdal
import numpy as np
import glob
import os
from PIL import Image
import time
import sys
import getopt
import tarfile
import json
import logging
import pathlib
from datetime import datetime as dt
#from gdalconst import *
from scipy import ndimage


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


def creategeotiff(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], array.shape[2], dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection()) 

    for i in range(0, array.shape[2]):
        dataSet.GetRasterBand(i+1).WriteArray(array[:, :, i])
    dataSet.FlushCache()
    
    return name

def readLandsatTarGz(in_zip, bands, sensor, cloudMasks="", shadowMask=""):
    flist = glob.glob(in_zip + '/*.tif')
    output_img = []

    for i in range(len(bands)):
        band = bands[i]
        chn_fn = None
    
        for fname in flist:  # CHANGED tif 
            if fname.find(band) > -1 and fname.endswith('.TIF') and 'T1' in fname :  # L2A
                #chn_fn = "/vsitar/%s/%s" % (in_zip,fname)                
                chn_fn = fname
                break
        if chn_fn is None: raise ValueError('Cannot find channel name in zip file: {b}'.format(b=band))

        ds = gdal.Open(chn_fn, 0)
        band_ds = ds.GetRasterBand(1)
        data = (band_ds.ReadAsArray()).astype(np.uint16)
        nl = data.shape[0]
        nc = data.shape[1]

        if 'L7' in sensor:   # fill missing stripes Landsat 7  without reconstructing clouds and shadows
            # Do only once at the beginning of the loop
            if i == 0:
                # read cloud and shadow masks for later filling missing stripes of Landsat 7
                cloud_shadow_mask = getOverallMask([cloudMasks,shadowMask])

                # TODO: define a better save_path for cloudy_stripes_mask
                cloudy_stripes_mask = getCloudyStripesMask(ds, in_zip+'/cloudy_stripes_mask.TIF', cloud_shadow_mask)
                
                overall_mask = cloud_shadow_mask + cloudy_stripes_mask
            
            data = fillStripesLandsat(ds, chn_fn, overall_mask)

        band = data[data!=band_ds.GetNoDataValue()]
        maxVal = np.quantile( band, 0.999)
        minVal = np.quantile( band, 0.001 )         
        data[(data > maxVal)*(data!=band_ds.GetNoDataValue())] = maxVal
        data[(data < minVal)*(data!=band_ds.GetNoDataValue())] = minVal
        output_img.append(data)
    
    output_img = np.array(output_img).astype(np.uint16)
    output_img = np.moveaxis(output_img, 0, -1)
    
    return output_img, ds


def getOverallMask(mask_path_list):
    if mask_path_list == []:
        return None
    ds_list = [gdal.Open(path) for path in mask_path_list]
    mask_list = [ds.GetRasterBand(1).ReadAsArray().astype(bool) for ds in ds_list]
    return np.sum(mask_list, axis=0, dtype=bool)


def getCloudyStripesMask(dataSet, save_path, cloud_shadow_mask):
    # Note that one could simply dilate enough the cloud-shadow mask in order to find the cloudy parts
    # of stripes, and then erode (i.e., closing). However the dilation factor would be too big and worsen
    # the cloud-shadow masks. Instead, explicitly estimating the cloudy stripes allow to use less dilation
    # (or no dilation) on the cloud-shadows mask later in the function 'fillStripesLandsat'.
    band_ds = dataSet.GetRasterBand(1)
    data = (band_ds.ReadAsArray()).astype(np.uint16)
    # Compute the stripes mask from no data areas of the image
    stripes_mask = data==band_ds.GetNoDataValue()

    # Dilate the cloud and shadow mask to connect masked areas split by the stripes
    dil_cloud_shadow_mask = ndimage.binary_closing(cloud_shadow_mask, iterations=10)

    # obtain only the cloudy parts of the stripes
    cloudy_stripes_mask = stripes_mask*dil_cloud_shadow_mask

    # delete small lines that are likely to be wrong
    cloudy_stripes_mask = ndimage.binary_opening(cloudy_stripes_mask, iterations=1)
    
    # Save the new mask to drive
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.CreateCopy(save_path,dataSet,strict=0)
    new_band_ds = new_ds.GetRasterBand(1)
    new_band_ds.WriteArray(cloudy_stripes_mask)
    
    return cloudy_stripes_mask


def fillStripesLandsat(dataSet, img_path, mask):
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.CreateCopy(img_path[:-4]+'_stripes-filled.TIF',dataSet,strict=0)
    new_band_ds = new_ds.GetRasterBand(1)
            
    # stripes filling
    gdal.FillNodata(targetBand=new_band_ds, maskBand=None, maxSearchDist=15, smoothingIterations=0)

    # This is used to deal with bad areas like ones with high density of clouds and shadows
    # that are small compared to the stripes.
    mask = ndimage.binary_closing(mask, iterations=5)
    keep_mask = mask^1

    # cloud and shadows masking
    data = new_band_ds.ReadAsArray()*keep_mask
    new_band_ds.WriteArray(data)
    return data.astype(np.uint16)
    
    
def readS2zip(in_zip, bands, tile):
	

    flist = []  
    
    for root, dirs, files in os.walk(in_zip):
	    for file in files:
            #append the file name to the list
		    flist.append(os.path.join(root,file))
    
    output_img = np.zeros(((10980,10980, len(bands))))
    dataSorce = None
    
    for i in range(len(bands)):
        band = bands[i]
        
        if band == 'B02' or band == 'B03' or band == 'B04' or band == 'B08':
            res = 10 
        elif band == 'B05' or band == 'B06' or band == 'B07' or band == 'B8A' or band == 'B11' or band == 'B12' or band == 'SCL':
    	    res = 20
        elif band == 'B01' or band == 'B09':
            res = 60 
        else:
    	    raise ValueError('Bad band values: {b}'.format(b=band))
    
        chn_fn = None
    

        for fname in flist:
            if fname.find(band) > -1 and 'IMG_DATA' in fname and fname.endswith('.jp2') and '%dm'%res in fname and '_T'+tile in fname:  # L2A
                chn_fn = fname   # % (in_zip,fname)                
                break
        if chn_fn is None: raise ValueError('Cannot find channel name in zip file: {b}, {r}, tile={t}'.format(b=band, r=res, t=tile))
        
        
        
        ds = gdal.Open(chn_fn, 0)
        band_ds = ds.GetRasterBand(1)
        data = (band_ds.ReadAsArray()).astype(np.uint16)
        
        if band == 'B02':
            dataSorce = ds
        if res != 10:
            data   = np.array(Image.fromarray(data).resize((10980, 10980), Image.NEAREST))
        
        band =  np.reshape(data, (10980*10980, 1))          
        maxVal = np.quantile( band, 0.999)
        minVal = np.quantile( band, 0.001 )         
        band[band > maxVal] = maxVal
        band[band < minVal] = minVal
        data = np.reshape(band, (10980, 10980))
        output_img[:,:,i] = data
        
    return output_img, dataSorce


def unzipS2andSaveToTif(zip_path, tif_path,  tile):
    bands = ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12']
    output_img, dataSorce = readS2zip(zip_path, bands, tile)
    name = os.path.basename(zip_path)[4:-5]
    output_path = os.path.join(tif_path, name + '.tif')
    creategeotiff(output_path, output_img, 0, dataSorce, gdal.GDT_UInt16)
      
    
        
def unzipLandsatandSaveToTif(zip_path, tif_path, sensor, cloudMasks='', shadowMask=''):
#    if 'L8' in sensor:
#            bands = ['band2','band3','band4','band5','band6', 'band7']
#    else:
#            bands = ['band1','band2','band3','band4','band5','band7']
            
    if 'L8' in sensor:
            bands = ['B2','B3','B4','B5','B6', 'B7']
    else:
            bands = ['B1','B2','B3','B4','B5','B7']

    output_img, dataSorce = readLandsatTarGz(zip_path, bands, sensor, cloudMasks, shadowMask)
    name = os.path.dirname(zip_path)
    output_path = os.path.join(tif_path, name + '.tif')
    creategeotiff(output_path, output_img, 0, dataSorce, gdal.GDT_UInt16)
    
       
    
def Help():
    print("\nThis script unzip and saves bands from original S2 L2A.zip. or Landsat tar.gz. to tiff."
          " For Sentinel 2 the bands with 20m spatial resolution are interpolated to 10m.\n"\
          "Required parameters :\n"\
          "  -i, --inputPath=     the path to the zip/tar.gz. file to be preprocessed \n"\
          "  -o, --outputPath=    the path where the processed tif will be saved \n"\
          "  -t, --tile=          the tile to be processed (specify only for S2)\n"\
          "  -d, --sensor=        the sensor type the possible values : S2, L5, L7, L8 \n"
          "  --cloud-mask=        the path to the cloud mask tiff file \n"\
          "  --shadow-mask=       the path to the shadow mask tiff file \n"
          )
    sys.exit(2)




if __name__ == "__main__":
##    try:
##      opts, args = getopt.getopt(sys.argv[1:],"t:i:o:d:",["help","tile=","inputPath=","outputPath=",
##                                                          "sensor=","cloud-mask=","shadow-mask="])
##    except getopt.GetoptError:
##      print('Invalid argument!')
##      sys.exit(2)
##    for opt, arg in opts:
##      if opt in ("--help"):
##         Help()
##      elif opt in ("-t", "--tile"):
##         tile = arg
##      elif opt in ("-i", "--inputPath"):
##         zip_path = arg
##      elif opt in ("-o", "--outputPath"):
##         tif_path = arg
##      elif opt in ("-d", "--sensor"):
##         sensor = arg
##      elif opt in ("--cloud-mask"):
##         cloud_mask_path = arg
##      elif opt in ("--shadow-mask"):
##         shadow_mask_path = arg
##    if (not opts):
##      Help()
##    #Log and configs init
##    config_file = "configuration_processor.json"
##    with open(config_file) as json_file:
##        config = json.load(json_file)
##
##    #--------------------------------------------------------------------------
##    #--- LOAD CONFIGURATION
##    #--------------------------------------------------------------------------
##    verbose = config["verbose"]
##    log_level = config["log_level"]
##
##    #--------------------------------------------------------------------------
##    #--- LOG SETUP
##    #--------------------------------------------------------------------------
##
##    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
##    logging.getLogger().setLevel(logging.NOTSET)
##    # log_filename.mkdir(parents=True, exist_ok=True)
##    output_path =pathlib.Path("/output")
##    log_filename = output_path / 'log_{}.log'.format(dt.now().strftime("%Y-%m-%d-%H-%M"))
##
##
##    # Add stdout handler, with level INFO
##    console = logging.StreamHandler(sys.stdout)
##    console.setLevel(logging.getLevelName(log_level.upper()))
##    formatter_1 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
##    console.setFormatter(formatter_1)
##    logging.getLogger().addHandler(console)
##
##    # Add file rotating handler, with level DEBUG
##    fileHandler = logging.FileHandler(filename=log_filename)
##    fileHandler.setLevel(logging.getLevelName(log_level.upper()))
##    formatter_2 = logging.Formatter('%(asctime)s [%(name)s-%(levelname)s]: %(message)s')
##    fileHandler.setFormatter(formatter_2)
##    logging.getLogger().addHandler(fileHandler)
##
##    log = logging.getLogger(__name__)
##    log.info("Configuration Complete.")
##
##    #--------------------------
##    #----- END LOGSETUP
##    #--------------------------
##    log.info("Start processing")
    gdal.PushErrorHandler(gdal_error_handler)
    sensor = "L7"
    zip_path = "2.opt-spectral-filtering_test-data/LE07_L2SP_166052_20060327_20200914_02_T1_orig"
    tif_path = "2.opt-spectral-filtering_test-data/LE07_L2SP_166052_20060327_20200914_02_T1_orig"
    tile = '42WXS'
    cloud_mask_path = ""
    shadow_mask_path = ""
    if sensor == "S2":
        unzipS2andSaveToTif(zip_path, tif_path,  tile)
    else:
        unzipLandsatandSaveToTif(zip_path, tif_path, sensor, cloud_mask_path, shadow_mask_path)
##    log.info("Processing complete")

