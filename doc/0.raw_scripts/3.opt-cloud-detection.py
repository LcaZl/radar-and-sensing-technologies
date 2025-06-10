# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:40:59 2020

@author: RSLab
"""

from osgeo import gdal
##import gdal
import numpy as np
import glob
import os
import zipfile
from PIL import Image
import sys
import getopt
import tarfile
import json
import logging
import pathlib
from datetime import datetime as dt
from sklearn.cluster import KMeans
from scipy.signal import medfilt2d


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




def readCloudShadowMask(in_zip, cloud, tile):
	
    flist = []  
    
    for root, dirs, files in os.walk(in_zip):
	    for file in files:
            #append the file name to the list
		    flist.append(os.path.join(root,file))

    res = 20
    chn_fn = None
    
    for fname in flist:
        if fname.find(cloud[0]) > -1 and 'IMG_DATA' in fname and fname.endswith('.jp2') and '%dm'%res in fname and '_T'+tile in fname:  # L2A
            #chn_fn = "/vsizip/%s/%s" % (in_zip,fname)  
            chn_fn =  str(fname)           
            break
    if chn_fn is None: raise ValueError('Cannot find channel name in zip file: {b}, {r}, tile={t}'.format(b=cloud, r=res, t=tile))
        
    ds = gdal.Open(chn_fn, 0)
    band_ds = ds.GetRasterBand(1)
    data = (band_ds.ReadAsArray()).astype(np.uint16)
    cloud_mask   = np.array(Image.fromarray(data).resize((10980, 10980), Image.NEAREST))
    nodata_mask      = cloud_mask==0
    cloud_mask_final = np.logical_or(cloud_mask==9, cloud_mask==10).astype(int)
    cloud_mask_final = np.logical_or(cloud_mask_final==1, cloud_mask==8).astype(int)
    shadow_mask_final = (cloud_mask==3).astype(int)
    geoTransform = ds.GetGeoTransform()
    newGeoTransform = [geoTransform[0], 10, geoTransform[2], geoTransform[3], geoTransform[4], -10]
    ds.SetGeoTransform(newGeoTransform)
   
    return cloud_mask_final, shadow_mask_final, nodata_mask, ds    

def unzipS2andSaveToTif(zip_path, cloudPath, shadowPath, nodataPath,  tile):
    cloud = ['SCL']
    name = os.path.basename(zip_path)[4:-5]
    cloud_mask, shadow_mask, nodata_mask, dataSorce = readCloudShadowMask(zip_path, cloud, tile)
    output_path_cloud = cloudPath  + '/' + name + '_cloudMediumMask_Sen2Cor.tif'
    creategeotiffOneBand(output_path_cloud, cloud_mask, 0, dataSorce, gdal.GDT_Byte)
        
    output_path_shadow = shadowPath  + '/' + name + '_shadowMask_Sen2Cor.tif'
    creategeotiffOneBand(output_path_shadow, shadow_mask, 0, dataSorce, gdal.GDT_Byte)
    
    output_path_nodata = shadowPath  + '/' + name + '_nodataMask_Sen2Cor.tif'
    creategeotiffOneBand(output_path_nodata, nodata_mask, 0, dataSorce, gdal.GDT_Byte)


def creategeotiffOneBand(name, array, NDV, dataSorce, dataType):
    
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = NDV
    dataSet = driver.Create(name, array.shape[1], array.shape[0], 1, dataType)
    dataSet.SetGeoTransform(dataSorce.GetGeoTransform())
    dataSet.SetProjection(dataSorce.GetProjection())
    dataSet.GetRasterBand(1).WriteArray(array[:, :])
    dataSet.FlushCache()
    
    return name

def loadTimeSeries2D(n_imgs,pix_region, patch_size, cloudMasks, shadowMasks):
    
    ts_2D = np.zeros((patch_size* patch_size, len(n_imgs)))
    for i in range(len(n_imgs)):   
            cmask_ds = gdal.Open(cloudMasks[i])
            cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)   
            
            smask_ds = gdal.Open(shadowMasks[i])
            smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            cloud_shadow_mask = (cmask + smask)
            #cloud_shadow_mask = cmask 
            
            not_cloud_shadow_mask = cloud_shadow_mask^1
            
            img_temp_ds = gdal.Open(n_imgs[i])
            img_temp= np.zeros((patch_size, patch_size))
            
            img_temp[:, :] = img_temp_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)    
            img_temp[:,:] = np.multiply(img_temp[:,:], not_cloud_shadow_mask)
                    
            img_temp[img_temp==0]=['nan']       
            img_temp =  np.reshape(img_temp, (patch_size*patch_size))
           
            ts_2D[:, i] = img_temp
            
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    
    return ts_2D


def getBackgroundImage(input_path, output_path,  year, cloudPath, shadowPath ): 
    
    cloudMasks_all =  sorted(glob.glob(cloudPath + '/*cloudMediumMask_Sen2Cor.tif'))   
    shadowMasks_all =  sorted(glob.glob(shadowPath + '/*shadowMask_Sen2Cor.tif'))   
    img_all =  sorted(glob.glob(input_path + '/*.tif'))  
    
    seasons = []
    seasons.append(['01', '02', '03']) # jan feb march  
    seasons.append(['04', '05', '06']) # april may ...
    seasons.append(['07', '08', '09'])
    seasons.append(['10', '11', '12'])

    for s in range(len(seasons)):  
        season = seasons[s]
        
        cloudMasks = sorted([x for x in cloudMasks_all if year+season[0] in x or year+season[1] in x or year+season[2] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year+season[0] in x or year+season[1] in x or year+season[2] in x ])
        season_img = sorted([x for x in img_all if year+season[0] in x or year+season[1] in x or year+season[2] in x ])
        
         
        if len(shadowMasks) != len(cloudMasks) or len(season_img) != len(cloudMasks):
            raise IOError("The number of clouds and shadow masks should coresponds to the number of Sentinel images!")
       
        if len(shadowMasks) != 0:
            target_ds = gdal.Open(season_img[0])
            nl=np.int(target_ds.RasterYSize)
            nc=np.int(target_ds.RasterXSize)
            #nb = target_ds.RasterCount      
            
            sesonalBackground = np.zeros((nl, nc))
            patch_sizeX = 2196 
            patch_sizeY = 2196                
            for i in range(5): 
                for j in range(5): 
                    pix_region = np.array([i*patch_sizeX, j*patch_sizeX])
                    img3d = loadTimeSeries2D(season_img, pix_region, patch_sizeX, cloudMasks, shadowMasks)
                    
                    #compute nanquantile - the faster approach
                    arr = img3d
                    mask = (arr >= np.nanmin(arr)).astype(int)
    
                    count = mask.sum(axis=1)
                    groups = np.unique(count)
                    groups = groups[groups > 0]
                    
                    temp = np.zeros((arr.shape[0]))
                    for g in range(len(groups)):
                        pos = np.where (count == groups[g])
                        values = arr[pos]
                        values = np.nan_to_num(values, 0) #nan=(np.nanmin(arr)-1))
                        values = np.sort (values, axis=1)
                        values = values[:,-groups[g]:]
                        temp[pos] = (np.percentile(values, 25, axis=1,interpolation='midpoint')) 
                        
                    sesonalBackground[j*patch_sizeX:j*patch_sizeX+patch_sizeX, i*patch_sizeX:i*patch_sizeX+patch_sizeX] =  np.reshape(temp ,[patch_sizeX,patch_sizeY])

            
            creategeotiffOneBand(os.path.join(str(output_path),'backgroundImage_' + tile + year + '_' + str(s+1) + '.tif'), sesonalBackground, 0, target_ds, gdal.GDT_Float32)
            target_ds = None
            del target_ds
       
        
        
def getCloudMask(input_path, finalCloudMaskPath,  year, input_path_cloud, backgroundPath ):
    
    mean =np.zeros(3)
    mask =np.zeros(3)
    dist =np.zeros(3)
        
    background_ds = gdal.Open(backgroundPath)
    background_img = background_ds.GetRasterBand(1).ReadAsArray(0, 0, 10980, 10980)
              
    finalMask = np.zeros((10980, 10980))
    img_temp_ds = gdal.Open(input_path)
    nl=np.int(img_temp_ds.RasterYSize)
    nc=np.int(img_temp_ds.RasterXSize)
    img_temp = img_temp_ds.GetRasterBand(1).ReadAsArray(0, 0, nl, nc)
            
    cloudMask_ds = gdal.Open(input_path_cloud)
    cloudMask = cloudMask_ds.GetRasterBand(1).ReadAsArray(0, 0, nl, nc)
    meanCloudMask = (cloudMask*img_temp).astype(float)
    meanCloudMask[meanCloudMask == 0] = ['nan']  
    meanCloudMask = np.nanmean(meanCloudMask);           
             
    diff = abs(img_temp - background_img )  # I added ABS!!!            
    sizeX = 10980 # 1000 #2196
    sizeY = 10980 #1000 #2196                                          
    num = np.unique(cloudMask, return_counts=True)[1]
    cloudPerc = num[1]/ (num[0]+num[1])
                                     
    if  cloudPerc > 0.0008:  
                        
        diff = diff.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(diff)
        cluster = np.reshape(kmeans.labels_, [sizeX,sizeY])
        diff = diff.reshape(sizeX, sizeY)
        
        for c in range(3):
            mask = ([cluster == c][0])
            masked = diff*mask
            masked[masked == 0] = ['nan']  
                            
            mean[c] = np.nanmean(masked)
            dist[c] = np.abs(meanCloudMask - mean[c])
                        
        ix = np.argsort(dist)    # np.argmin(dist)
        mask = ([cluster == ix[0]][0])
        #mask2 = ([cluster == ix[1]][0])
        finalMask = mask.astype(int) + cloudMask #+mask2.astype(int)
    else:
        finalMask = cloudMask
                        
    finalMask[finalMask>1] = 1
    creategeotiffOneBand(finalCloudMaskPath, finalMask, 0, cloudMask_ds, gdal.GDT_Byte)

    img_temp_ds = None
    cloudMask_ds = None
    del img_temp_ds
    del cloudMask_ds
    
       
def getShadowdMask(input_path, finalShadowMaskPath,  year, finalCloudMaskPath, shadowPath ):
  
    img_temp_ds = gdal.Open(input_path)
    sizeX=np.int(img_temp_ds.RasterYSize)
    finalShadowMask = np.zeros((sizeX, sizeX))

    img_patch = img_temp_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)
    cloudMask_ds = gdal.Open(finalCloudMaskPath)
    cloud = cloudMask_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)
    shadow_ds = gdal.Open(shadowPath)
    shadow = shadow_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)

    meanCloudMask_patch = (cloud*img_patch).astype(float)
    meanCloudMask_patch[meanCloudMask_patch == 0] = ['nan']  
    meanCloudMask_patch = np.nanmean(meanCloudMask_patch) # do we need it 
    num = np.unique(cloud, return_counts=True)[1]
    cloudPerc = num[1]/ (num[0]+num[1])
                    
    if cloudPerc > 0.0008 and meanCloudMask_patch > 1400:
        blue = (img_temp_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)).astype(np.float)
        nir = (img_temp_ds.GetRasterBand(7).ReadAsArray( 0, 0, sizeX, sizeX)).astype(np.float)
        swir11 = (img_temp_ds.GetRasterBand(9).ReadAsArray(0, 0, sizeX, sizeX)).astype(np.float)

        blue[cloud == 1] = ['nan'] 
        nir[cloud == 1] = ['nan']  
        swir11[cloud == 1] = ['nan']  
        csi = (nir+swir11)/2
                    
        t1 = np.nanmin(csi) + 0.5*(np.nanmean(csi) - np.nanmin(csi))
        t2 = np.nanmin(blue) + 0.25*(np.nanmean(blue) - np.nanmin(blue))          
        csi_th = csi<=t1
        blue_th = blue<=t2
        mask = (csi_th*blue_th).astype(np.uint8);
        finalShadowMask = medfilt2d(mask, kernel_size=3)
        finalShadowMask = finalShadowMask + shadow
    else:
        finalShadowMask = shadow
             
    finalShadowMask[finalShadowMask>1] = 1
    creategeotiffOneBand(finalShadowMaskPath, finalShadowMask, 0, cloudMask_ds, gdal.GDT_Byte)

    img_temp_ds = None
    cloudMask_ds = None
    shadow_ds = None
    del img_temp_ds
    del cloudMask_ds
    del shadow_ds



def readCloudShadowMaskLandsat(zip_path):
    flist = glob.glob(zip_path + '/*.tif')
    
    for fname in flist:   #'_pixel_qa'   '_QA_PIXEL'   '.tif'   '.TIF'
        if '_QA_PIXEL' in fname  and (fname.endswith('.TIF') or fname.endswith('.tif')) :  # L2A
            #chn_fn = "/vsitar/%s/%s" % (zip_path,fname)
            chn_fn = fname
            break
    if chn_fn is None: raise ValueError('Cannot find cloud channel name in zip file: {b}'.format(b=zip_path))
        
    ds = gdal.Open(chn_fn, 0)
    band_ds = ds.GetRasterBand(1)
    img_qa = (band_ds.ReadAsArray()).astype(np.uint16)
    img_qa_values = np.unique(img_qa)                                   # QA values
    mask_qa_values = np.zeros(len(img_qa_values), dtype=np.uint8)       # QA corresponding mask values
    binary_qa_values = []  

    
    for i in range(len(img_qa_values)):
        b = np.binary_repr(img_qa_values[i], width=16)
        binary_qa_values.append(b)
        
        #SNOW
        if b[-6] == '1':
            mask_qa_values[i] = 5
        
        #CLOUD SHADOWS
        if b[-5] == '1' or (b[-11] == '1' and b[-12] == '1'):
            mask_qa_values[i] = 4
        
        #CLOUDS: medium-prob and high-prob cloud and cirrus
        if (b[-3] == '1' or b[-4] == '1' or (b[-9] == '0' and b[-10] == '1') or (b[-9] == '1' and b[-10] == '1') or (b[-15] == '1' and b[-16] == '1')):          
            mask_qa_values[i] = 3
            
        #FILL (NAN)
        if b[-1] == '1':
            mask_qa_values[i] = 2
    #MASK
    height, width = img_qa.shape
    mask = np.zeros( (height,width), dtype=np.uint8)
    
    for i in range(len(img_qa_values)):
        layer = (img_qa == img_qa_values[i])
        mask[layer] = mask_qa_values[i]
        
    mask = mask.astype(int)
    shadow_mask = mask == 4
    cloud_mask = mask == 3
    snow_mask = mask == 1
    
    nodata_mask = mask == 2 # NEW GP

#    return  cloud_mask, shadow_mask, snow_mask, ds  
    return  cloud_mask, shadow_mask, nodata_mask, snow_mask, ds # NEW GP



def readSnowMaskLandsat(in_zip, snow):
	
    if not os.path.isfile(in_zip):
	    raise IOError("Input tar.gz. doesn't exists")

    try:
        tf= tarfile.open(in_zip)
        flist = tf.getnames();
        tf.close()
    except:
	    raise IOError("Unable to open tar.gz-file.")
		
    for fname in flist:
        if '_pixel_qa' in fname  and fname.endswith('.tif') :  # L2A
            chn_fn = "/vsitar/%s/%s" % (in_zip,fname)                
            break
    if chn_fn is None: raise ValueError('Cannot find cloud channel name in zip file: {b}'.format(b=in_zip))
        
    ds = gdal.Open(chn_fn, 0)
    band_ds = ds.GetRasterBand(1)
    band = (band_ds.ReadAsArray()).astype(np.uint16)
    snow_mask = np.isin(band, snow).astype(int)
    
    return snow_mask   



def unzipLandsatandSaveToTif(zip_path, cloudPath, shadowPath, nodataPath, sensor, area):

    name = os.path.dirname(zip_path)
    name = os.path.basename(name)
    cloud_mask, shadow_mask, nodata_mask, snow_mask,  dataSorce = readCloudShadowMaskLandsat(zip_path)
    
    if area == "Siberia" or area == "siberia":
       # snow_mask = readSnowMaskLandsat(zip_path, snow)
        output_path_cloud = os.path.join(cloudPath, name + '_snowIceMask.tif')
        creategeotiffOneBand(output_path_cloud, snow_mask, 0, dataSorce, gdal.GDT_Byte)

    output_path_cloud = os.path.join(cloudPath, name + '_cloudMediumMask.tif')
    creategeotiffOneBand(output_path_cloud, cloud_mask, 0, dataSorce, gdal.GDT_Byte)
    
    output_path_shadow = os.path.join(shadowPath, name + '_shadowMask.tif')
    creategeotiffOneBand(output_path_shadow, shadow_mask, 0, dataSorce, gdal.GDT_Byte)
    
    output_path_nodata = os.path.join(shadowPath, name + '_nodataMask.tif')
    creategeotiffOneBand(output_path_nodata, nodata_mask, 0, dataSorce, gdal.GDT_Byte)



def getSeason(img_month):
    
    seasons = []   
    seasons.append(['01', '03'])
    seasons.append(['04', '06'])
    seasons.append(['07', '09'])
    seasons.append(['10', '12'])
    for m in range(len(seasons)):
        season = seasons[m]
        if season[0] <= img_month <= season[1]:
            final_season = m +1
            return final_season
        
        
    
def Help():
    print("\nThis script reads the cloud mask and shadow mask from original S2 L2A.zip. or Landsat tar.gz. \n"\
          "\n For the Sentinel-2 data: first  run this scipt with b=0, than with b=1 and finally with b=2 \n"\
          "Required parameters :\n"\
          "  -i, --inputPath=     the path to the zip file to be preprocessed\n"\
          "  -f,                  the path to the tif file to be preprocessed (only for Sentinel 2)\n"\
          "  -c, --cloudPath=     the output path where the cloud masks will be saved \n"\
          "  -s, --shadowPath=    the output path where the shadow masks will be saved \n"\
          "  -t, --tile=          the tile to be processed (specify only for S2)\n"\
          "  -d, --sensor=        the sensor type the possible values : S2, L5, L7, L8 \n"\
          "  -a,                  the area to be processed (Africa, Amazonia or Siberia))"\
          "  -y,                  the year for which four composites will be created \n"\
          "  -b                   the flag,  assaign 1 to run script only to produce 4 seasonal background images (for this option define -g nad -o), assaign 1  \n"\
          "  -g,                  the path to the all tif files for considered Sentinel 2 tile to compute background image (only for Sentinel 2)\n"\
          "  -o,                  the path where 4 seasonal background images will be saved\n")
    sys.exit(2)

  
if __name__ == "__main__":   
##    background = 0
##    try:
##      opts, args = getopt.getopt(sys.argv[1:],"t:i:f:g:o:c:s:d:a:y:b:",["help","tile=","inputPath=","cloudPath=","shadowPath=", "sensor="])
##    except getopt.GetoptError:
##      print('Invalid argument!')
##      sys.exit(2)
##    for opt, arg in opts:
##      if opt in ("--help"):
##        Help()
##      elif opt in ("-t", "--tile"):
##        tile = arg
##      elif opt in ("-i"):
##        input_zip = arg
##      elif opt in ("-f"):
##        input_tif = arg
##      elif opt in ("-g"):
##        input_background = arg
##      elif opt in ("-o"):
##        output_path = arg 
##      elif opt in ("-c", "--cloudPath"):
##        cloudPath = arg
##      elif opt in ("-s", "--shadowPath"):
##        shadowPath = arg
##      elif opt in ("-d", "--sensor"):
##        sensor = arg
##      elif opt in ("-a"):
##          area = arg
##      elif opt in ("-y"):
##           year = arg
##      elif opt in ("-b"):
##          background = arg
##    if (not opts):
##      Help()
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
    background = 0
    input_zip = "2.opt-spectral-filtering_test-data/LE07_L2SP_166052_20060327_20200914_02_T1_orig"
    cloudPath = "3.opt-cloud-detection_test-data/cloud_masks"
    shadowPath = "3.opt-cloud-detection_test-data/shadow_masks"
    nodataPath = "3.opt-cloud-detection_test-data/nodata_masks"

    input_tif = '3.opt-cloud-detection_test-data/MSIL2A_20190731T065629_N0213_R063_T42WXS_20190731T120318.tif'
    area = "Siberia"
    input_zip = "2.opt-spectral-filtering_test-data/S2B_MSIL2A_20190704T070629_N0212_R106_T42WXS_20190704T104800.SAFE"
    sensor = "S2"
    input_background = '3.opt-cloud-detection_test-data'
    tile = '42WXS'
    output_path = '3.opt-cloud-detection_test-data/seasonal-background'
    year = '2019'


##    background = 0
##    input_zip = "2.opt-spectral-filtering_test-data/LE07_L2SP_166052_20060327_20200914_02_T1_orig"
##    cloudPath = "3.opt-cloud-detection_test-data/cloud_masks"
##    shadowPath = "3.opt-cloud-detection_test-data/shadow_masks"
##    nodataPath = "3.opt-cloud-detection_test-data/nodata_masks"
##    sensor = "L7"
##    area= "africa"
    if background == 1: # this condition should be run once for each S2 tile ( after Sen2cor mask has been read)
        
        getBackgroundImage(input_background, output_path,  year, cloudPath, shadowPath )
        
    elif background == 2: # this condition should be run once for each S2 image (after Sen2cor mask and backgorund image have been read)    
        #generates cloud and shadow path correspodning to input_tif 
        img_name = os.path.basename(input_zip)[4:-5]
        s2_path_cloud = os.path.join(cloudPath, img_name + '_cloudMediumMask_Sen2Cor.tif')
        s2_path_shadow = os.path.join(shadowPath, img_name + '_shadowMask_Sen2Cor.tif')
        finalCloudMaskPath =  s2_path_cloud[:-12] + '.tif'
        finalShadowMaskPath =  s2_path_shadow[:-12] + '.tif'
               
        #generates correspodning backgroundImage path
        img_month = img_name[img_name.index(year)+4:img_name.index(year)+6]
        season = getSeason(img_month)
        backgroundPath = os.path.join(output_path, 'backgroundImage_' + tile + year + '_' + str(season) + '.tif')


        getCloudMask(input_tif, finalCloudMaskPath,  year, s2_path_cloud, backgroundPath )
        getShadowdMask(input_tif, finalShadowMaskPath,  year, finalCloudMaskPath, s2_path_shadow)
    
    else: # background == 0
        if sensor == "S2": #this step read Sen2cor mask
            unzipS2andSaveToTif(input_zip, cloudPath, shadowPath, nodataPath, tile)
        else:
            unzipLandsatandSaveToTif(input_zip, cloudPath, shadowPath, nodataPath, sensor, area)      
##    log.info("Processing complete")


