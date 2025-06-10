# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:38:24 2021

@author: rslab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:25:14 2020

@author: RSLab
"""
import gdal
import numpy as np
import sys
import getopt
import json
import logging
import pathlib
from datetime import datetime as dt
import glob
import cv2
import rasterio
import os
from rasterio.fill import fillnodata


##############################################################################
# The band order of the output Sentinel-2 composites:
#    'B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12'

# The band order of the output Landsat 5/7 composites:
#    'band1','band2','band3','band4','band5','band7'

# The band order of the output Landsat 8 composites:
#    'band2','band3','band4','band5','band6', 'band7'
##############################################################################

def coord2pixel2(geo_transform, lat, long):
    xOffset = int((lat - geo_transform[2]) / geo_transform[0])
    yOffset = int((long - geo_transform[5]) / geo_transform[4])

    return xOffset, yOffset

def calculate_slope(DEM_path, output_path):
    gdal.DEMProcessing(output_path, DEM_path, 'slope')
    with rasterio.open(output_path) as dataset:
        slope=dataset.read(1)
    return slope

def calculate_aspect(DEM_path, output_path):
    gdal.DEMProcessing(output_path, DEM_path, 'aspect')
    with rasterio.open(output_path) as dataset:
        aspect=dataset.read(1)
    return aspect

def resizeImg(dem_path, output_path, tif_path):
    tif_ds = gdal.Open(tif_path)
    nl = np.int(tif_ds.RasterYSize)
    nc = np.int(tif_ds.RasterXSize)
    
    #coordinates where Sentinel2 starts
    geoTransform = tif_ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    
    with rasterio.open(dem_path) as src:   
         rows, cols = src.shape
         arr = src.read(1)
         arr_filled = fillnodata(arr, mask=src.read_masks(1), max_search_distance=10, smoothing_iterations=0)
         transform = src.transform
         
    #translates Sentinel2  coordinates to row and column of DEM image
    row, col = coord2pixel2(transform, minx, maxy)  
    dem_final = arr_filled[col:col+nc, row:row+nl]        
    creategeotiffOneBand(output_path, dem_final, 0, tif_ds, gdal.GDT_Int32)
    del tif_ds   

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
        
def restoreCloudsSavePixelByPixel(cloudMasks_path, shaowdMasks_path, tif_path, out_path, imgBefore_path = "", imgAfter_path = ""):
    
    cloud_ds = gdal.Open(cloudMasks_path)
    cloud_mask = np.asarray(cloud_ds.ReadAsArray(0, 0, np.int(cloud_ds.RasterXSize),np.int(cloud_ds.RasterYSize)))
    shadow_ds = gdal.Open(shaowdMasks_path)
    shadow_mask = np.asarray(shadow_ds.ReadAsArray(0, 0, np.int(shadow_ds.RasterXSize),np.int(shadow_ds.RasterYSize)))
    cloud_shadow_mask = cloud_mask + shadow_mask;
    pixMask = np.nonzero(cloud_shadow_mask==1)
    if len(pixMask[0]) == 0:  
           raise IOError(tif_path + " image does not have clouds!")
        
    imgToRestore_ds = gdal.Open(tif_path, 1)    
    imgToRestore = np.asarray(imgToRestore_ds.ReadAsArray(0, 0, np.int(imgToRestore_ds.RasterXSize),np.int(imgToRestore_ds.RasterYSize)))
    imgToRestore = np.moveaxis(imgToRestore, 0, -1)      
    src_ds_tab = []
    if not imgBefore_path:
            src_ds_tab.append(gdal.Open(imgAfter_path)) 
    elif not imgAfter_path:
            src_ds_tab.append(gdal.Open(imgBefore_path)) 
    else:
            src_ds_tab.append(gdal.Open(imgBefore_path)) 
            src_ds_tab.append(gdal.Open(imgAfter_path)) 
            

   #for each pixel in cloud-mask
    for i in range(int(len(pixMask[0]))):  
            px = int(pixMask[0][i])
            py = int(pixMask[1][i])
            pix_ts = np.zeros((len(src_ds_tab),10))
            for j in  range(int(len(src_ds_tab))): 
                temp = np.asarray(src_ds_tab[j].ReadAsArray(py, px, 1,1))
                pix_ts[j,:] = temp[:,0,0]
             
            for b in range(imgToRestore_ds.RasterCount):
                if len(src_ds_tab) == 1:
                    imgToRestore[px,py,b]  =  pix_ts[:,b]
                else:
                    imgToRestore[px,py,b] = (pix_ts[0,b] + pix_ts[1,b])/2

    creategeotiff(out_path, imgToRestore, 0, imgToRestore_ds, gdal.GDT_UInt16)    
    del imgToRestore_ds   

def creategeotiffOneBand(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], 1, dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection())
    dataSet.GetRasterBand(1).WriteArray(array[:, :])
    dataSet.FlushCache()
    
    return name

def restoreCloudsSavePixelByPixel2(tif_path, out_path, imgBefore_path = "", imgAfter_path = ""):
   
    imgToRestore_ds = gdal.Open(tif_path, 1)    
    imgToRestore = np.asarray(imgToRestore_ds.ReadAsArray(0, 0, np.int(imgToRestore_ds.RasterXSize),np.int(imgToRestore_ds.RasterYSize)))
    imgToRestore = np.moveaxis(imgToRestore, 0, -1) 
    pixMask = np.where(~imgToRestore.any(axis=2))

    
    #pixMask = np.nonzero(imgToRestore[:]==0)
    if len(pixMask[0]) != 0:          
        
        src_ds_tab = []
        if not imgBefore_path:
                src_ds_tab.append(gdal.Open(imgAfter_path)) 
        elif not imgAfter_path:
                src_ds_tab.append(gdal.Open(imgBefore_path)) 
        else:
                src_ds_tab.append(gdal.Open(imgBefore_path)) 
                src_ds_tab.append(gdal.Open(imgAfter_path)) 
                
    
       #for each pixel in cloud-mask
        for i in range(int(len(pixMask[0]))):  
                px = int(pixMask[0][i])
                py = int(pixMask[1][i])
                nb = imgToRestore_ds.RasterCount                  
    
                pix_ts = np.zeros((len(src_ds_tab), nb))
                for j in  range(int(len(src_ds_tab))): 
                    temp = np.asarray(src_ds_tab[j].ReadAsArray(py, px, 1,1))
                    pix_ts[j,:] = temp[:,0,0]
                 
                for b in range(imgToRestore_ds.RasterCount):
                    if len(src_ds_tab) == 1:
                        imgToRestore[px,py,b]  =  pix_ts[:,b]
                    else:
                        imgToRestore[px,py,b] = (pix_ts[0,b] + pix_ts[1,b])/2
                        

        creategeotiff(out_path, imgToRestore, 0, imgToRestore_ds, gdal.GDT_UInt16)    
        del imgToRestore_ds                     
        return out_path
    else:
        return tif_path

def getCompositeShadow(composite_path, output_path, year, slope_path):
    if int(year) == 2019:
        nir_band = 7  #Sentinel
        swir_band = 9
    else:
        nir_band = 4 #Landsat 
        swir_band = 5

    img_ds = gdal.Open(composite_path)
    nl=np.int(img_ds.RasterYSize)
    nc=np.int(img_ds.RasterXSize)
    nb = img_ds.RasterCount  
    
    blue = img_ds.GetRasterBand(1).ReadAsArray(0, 0, nc, nl)  
    nir = img_ds.GetRasterBand(nir_band).ReadAsArray(0, 0, nc, nl)
    swir11 = img_ds.GetRasterBand(swir_band).ReadAsArray(0, 0, nc, nl)    
    slope_ds = gdal.Open(slope_path)
    slope = slope_ds.GetRasterBand(1).ReadAsArray(0, 0, nc, nl)
   
    csi = ((nir + swir11)/2) # .astype(np.uint16)    
    t1 = (np.nanmin(csi) + 0.5*(np.nanmean(csi) - np.nanmin(csi)))
    t2 = (np.nanmin(blue) + 0.25*(np.nanmean(blue) - np.nanmin(blue)) )
    csi_th = csi<=t1
    blue_th = blue<=t2
    mask = (csi_th*blue_th).astype(np.uint8)
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # define a disk element
    mask_dil = cv2.dilate(mask,se,iterations = 1)

    # threshold for water removal check
    mask_dil = slope * mask_dil
    mask_dil =  mask_dil > 15  # threshold for water removal 
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # define a disk element
    mask_dil = cv2.dilate(mask_dil.astype(np.uint8),se,iterations = 1)
    mask_dil_inv = (mask_dil)^1  
    
    #creategeotiffOneBand(os.path.join('C:/Users/rslab/Desktop/new_composite/','mask_ringFd.tif'), mask_dil, 0, cloudMask_ds, gdal.GDT_Float32)
    s2_matched = np.zeros(((nl, nc, nb)))
        
    for b in range(nb):
            band = img_ds.GetRasterBand(b + 1).ReadAsArray(0, 0, nc, nl)   
            band_sh = (np.multiply(band, mask_dil)).astype(np.float32)
            band_sh[band_sh == 0] = ['nan'] 
            
            band_no_sh = np.multiply(band, mask_dil_inv).astype(np.float32)
            band_no_sh[band_no_sh == 0] = ['nan'] 
    
            mean_sh_band = np.nanmean(band_sh)
            stds_sh_band = np.nanstd(band_sh,  ddof=1)
            mean_no_sh_band = np.nanmean(band_no_sh)
            stds_no_sh_band = np.nanstd(band_no_sh,  ddof=1)
    
            par_b = stds_no_sh_band/stds_sh_band
            par_a = mean_no_sh_band - (par_b*mean_sh_band)
            band_sh_matched = np.rint((band_sh*par_b)+par_a)
            
            band_sh_matched[np.isnan(band_sh_matched)] = 1
            
            band_no_sh_matched = band_no_sh
            band_no_sh_matched[np.isnan(band_no_sh_matched)] = 1
            
            s2_matched[:,:,b] = band_sh_matched*band_no_sh_matched
        
    se= np.ones((3,3))
    mask_dil_dil = cv2.dilate(mask_dil,se,iterations = 1)
    mask_dil_ero = cv2.erode(mask_dil,se,iterations = 1)
    mask_ring = mask_dil_dil - mask_dil_ero

    s2_matched_fill = np.zeros(((nl, nc, nb)))
    for b in range(nb):
            band = s2_matched[:,:,b]
            s2_matched_fill[:,:,b] = cv2.inpaint(band.astype(np.float32), mask_ring, 3, cv2.INPAINT_TELEA)  # TYPE can be changed, size =3???
       
    s2_matched_fill2= s2_matched_fill.astype(np.uint16)
    creategeotiff(output_path, s2_matched_fill2, 0, img_ds, gdal.GDT_UInt16)

        
    blue = None
    nir =  None
    swir11 =  None
    band = None
    slope_ds = None
    img_ds = None
    del slope_ds
    del img_ds
    del band   
    del blue
    del nir
    del swir11
   
def Help():
    print("\nThis script reads the cloud mask and shadow mask from original S2 L2A.zip.\n"\
          "Required parameters :\n"\
          "  -i    the path to the preprocessed tif file (t) \n"\
          "  -o    the path where the processed tif will be saved \n"\
          "  -b    the path to the image one before the image to be restored (t-1), in case there is no image before leave it empty\n"\
          "  -n    the path to the image one after the image to be restored (t+1), in case there is no image after leave it empty\n"\
          "  -a    the area to be processed (Africa, Amazonia or Siberia))\n"\
          "  -y    the year for which four composites will be created \n"\
          "  -t    the tile of the Sentinel-2 image \n"\
          "  -d    the path to the DEM)\n"\
          "  -s    the path where the slope tif will be saved)\n"
          "  -p    the path where the aspect tif will be saved)\n")

    sys.exit(2)

  
if __name__ == "__main__":   

    try:
      opts, args = getopt.getopt(sys.argv[1:],"i:o:b:n:a:d:s:p:y:t:",["help"])
    except getopt.GetoptError:
      print('Invalid argument!')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("--help"):
          Help()

      elif opt in ("-i"):
          tif_path = arg
      elif opt in ("-o"):
          out_path = arg
      elif opt in ("-b"):
          imgBefore_path = arg
      elif opt in ("-n"):
          imgAfter_path = arg
      elif opt in ("-a"):
          area = arg
      elif opt in ("-d"):
          dem_path = arg
      elif opt in ("-s"):
          slope_output_path = arg
      elif opt in ("-p"):
          aspect_output_path = arg
      elif opt in ("-y"):
          year = arg
      elif opt in ("-t"):
          tile = arg
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
    try:
        imgAfter_path
    except NameError:
        imgAfter_path = ''
    try:
        imgBefore_path
    except NameError:
        imgBefore_path = ''

    log.info("Start processing")   
    
    if area == "Siberia" or area == "siberia":
        restored_path = tif_path
        #log.info("Siberia has a unique composite. By default we do not perform restoration.")  
    else: 
        restored_path = os.path.basename(out_path)[:-4] + '_temp.tif'
        restored_path = restoreCloudsSavePixelByPixel2(tif_path, restored_path, imgBefore_path, imgAfter_path)  
    
    #topographic shadow removal
    slope_path =  os.path.join(slope_output_path, str(tile) + str(year) + '_slope.tif')
    aspect_path =  os.path.join(aspect_output_path,  str(tile) + str(year) + '_aspect.tif')
    
    #cut  the DEM according to image extent
    output_dem_path =  dem_path[:-4] + "_final.tif"
    resizeImg(dem_path, output_dem_path, tif_path)
    slope = calculate_slope(output_dem_path, slope_path)
    aspect = calculate_aspect(output_dem_path, aspect_path)
    
    getCompositeShadow(restored_path, out_path, year, slope_path)

    log.info("Processing complete")
    
    

  

