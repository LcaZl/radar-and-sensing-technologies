"""
3.1 Cloud and Shadow detection

Cloud and cloud shadow detection are based on cloud and shadow mask provided with the Sen2Cor (for Sentinel-
2) and Fmask (for Landsat). The OA of cloud and shadow masks provided by the Sen2Cor (84%) is on average
lower than the one provided by Fmask (90%) [6]. Therefore, the Sen2cor masks should be further enhanced to
achieve the required accuracy. To this end, we adopt two strategies, one for cloud detection and one for cloud
shadow detection and removal.
For cloud detection, we compute the cloudless background image for each season The difference between the blue bands of 
each image from the TS and the background image is computed. The pixels in the difference image are then clustered into 3 clusters.
To understand which from the obtained clusters belong to cloud cover, the mean of each cluster is compared with the blue band mean of the cloudy pixels overall
image. Finally, we merge of the obtained cloud mask with the original Sen2cor mask. Note that this strategy is
performed only for tiles with a sufficiently large cloud cover, in order to properly model the clusters.
"""

import glob
import zipfile
from sklearn.cluster import KMeans
from scipy.signal import medfilt2d
from PIL import Image
import uuid
import logging
import os
from threading import Thread

from utility import *

def Help():
    logging.info("\nThis script reads the cloud mask and shadow mask from original S2 L2A.zip. or Landsat tar.gz. \n"\
          "\n For the Sentinel-2 data: first  run this script with b=0, than with b=1 and finally with b=2 \n"\
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

def readSentinelZipImage(zip_ref, zip_path, configuration, bands, resolution, verbose = False):
    """
    Reads and extracts the relevant Sentinel-2 image file from a zip archive.

    Parameters:
    - zip_file: Path to the Sentinel-2 zip file.
    - configuration: Configuration dictionary containing output paths and tile information.

    Returns:
    - Path to the extracted image file.
    """
    if verbose:
        logging.info(f"Searching {bands} at {resolution}m resolution in: {zip_path}")

    flist = [(s, s.split("/")[-1]) for s in zip_ref.namelist()]
    chn_fn = None
    paths = {}
    
    for archive_name, filename in flist:
            
        if any(band in filename for band in bands) and 'IMG_DATA' in archive_name and filename.endswith('.jp2') and f"_T{configuration['tile']}" in archive_name and '%dm'%resolution in filename:
            
            c_band = [band for band in bands if band in filename]

            if len(c_band) > 1:
                raise ValueError("Error. File belonging to multiple bands detected")
            
            c_band = c_band[0]
            chn_fn = (archive_name, filename)

            extracted_path = os.path.join(configuration["output_folder_path"], chn_fn[1])
            with zip_ref.open(chn_fn[0]) as source, open(extracted_path, 'wb') as target:
                shutil.copyfileobj(source, target)

            if not os.path.exists(extracted_path):
                raise RuntimeError(f"File not found after extraction: {extracted_path}")

            paths[c_band] = extracted_path
            
            if verbose:
                logging.info(f" -> File {chn_fn[1]} extracted at: {extracted_path}")
                
    if len(paths) == 0:
        raise ValueError(f"Cannot find appropriate image channel in zip file: {zip_path}")

   
    return paths

def resample_band(source_ds, target_ds, target_res):
    """
    Resample the source dataset to match the target resolution and extent.

    Parameters:
    - source_ds: GDAL dataset of the source band.
    - target_ds: GDAL dataset of the target resolution (reference).
    - target_res: Target resolution.

    Returns:
    - Resampled dataset.
    """
    unique_path = f"/vsimem/resampled_band_{uuid.uuid4().hex}.tif"
    gdal.Warp(
        unique_path,
        source_ds,
        width=target_ds.RasterXSize,
        height=target_ds.RasterYSize,
        resampleAlg="bilinear",  # Use bilinear resampling
    )
    return unique_path

########################################################
#   STAGE 1 - Extract initial cloud and shadow masks   #
########################################################

def readCloudShadowMask(configuration, cloud):
    """
    Extracts and processes cloud and shadow masks from a Sentinel-2 zip file.

    Parameters:
    - configuration: Application parameters.
    - cloud: List of cloud mask identifiers (e.g., 'SCL').

    Steps:
    1. Open the Sentinel-2 zip file.
    2. List all files in the zip archive and identify the appropriate cloud mask file based on the band, resolution, and tile.
    3. Extract the identified file to the specified output directory.
    4. Open the extracted file with GDAL and read the raster data.
    5. Resample the cloud mask to 10m resolution if necessary.
    6. Create and return the final cloud, shadow and no data masks along with the GDAL dataset object.
    7. Clean up by deleting the extracted file after processing.
    """
    logging.info("Reading Sentinel Cloud and Shadow masks ...")

    with zipfile.ZipFile(configuration["input_zip"], 'r') as zip_ref:

        # List all files in the zip
        flist = [(s, s.split("/")[-1]) for s in zip_ref.namelist()]
        chn_fn = None
        
        res = 20  # Sentinel-2 L2A SCL data has a resolution of 20 meters
        
        # Find the relevant .jp2 file for the cloud mask
        for (archive_name, filename) in flist:
            if filename.find(cloud[0]) > -1 and archive_name.find('IMG_DATA') > -1 and filename.endswith('.jp2') and archive_name.find('%dm' % res) > -1 and archive_name.find('_T' + configuration["tile"]) > -1:
                chn_fn = (archive_name, filename)
                break
        
        if chn_fn is None:
            raise ValueError('Cannot find channel name in zip file: {b}, {r}, tile={t}'.format(b=cloud, r=res, t=configuration["tile"]))

        logging.info(f" -> SCL file founded: {os.path.basename(chn_fn[0])}")

        # Extract the file to the specified output directory
        extracted_path = os.path.join(configuration["output_folder_path"], chn_fn[1])
        with zip_ref.open(chn_fn[0]) as source, open(extracted_path, 'wb') as target:
            shutil.copyfileobj(source, target)

        # Ensure the file was extracted successfully
        if not os.path.exists(extracted_path):
            raise RuntimeError(f"File not found after extraction: {extracted_path}")

        # Open the extracted file with GDAL
        ds = gdal.Open(extracted_path, 0)
        band_ds = ds.GetRasterBand(1)
        data = (band_ds.ReadAsArray()).astype(np.uint16)

        # Resample the cloud mask to 10m resolution if necessary
        cloud_mask = np.array(Image.fromarray(data).resize((10980, 10980), Image.NEAREST))
        nodata_mask = cloud_mask == 0  # Identify no data regions
        cloud_mask_final = np.logical_or(cloud_mask == 9, cloud_mask == 10).astype(int)  # Cloud masks
        cloud_mask_final = np.logical_or(cloud_mask_final == 1, cloud_mask == 8).astype(int)  # Additional cloud masks
        shadow_mask_final = (cloud_mask == 3).astype(int)  # Shadow mask

        # Update the GeoTransform to match 10m resolution
        geoTransform = ds.GetGeoTransform()
        newGeoTransform = [geoTransform[0], 10, geoTransform[2], geoTransform[3], geoTransform[4], -10]
        ds.SetGeoTransform(newGeoTransform)
    
    os.remove(extracted_path)  # Clean up by removing the extracted file

    return cloud_mask_final, shadow_mask_final, nodata_mask, ds  # Return the masks and dataset object


def unzipS2andSaveToTif(configuration):
    """
    Extracts cloud, shadow, and no data masks from Sentinel-2 zip files and saves them as GeoTIFFs.

    Parameters:
    - configuration: Application parameters containing input and output paths.

    Steps:
    1. Process each zip file matching the input pattern.
    2. Extract and process cloud and shadow masks.
    3. Save masks as GeoTIFF files in specified directories.
    """
    cloud = ['SCL']  # Define the band identifier for cloud masks

    # Get the list of all .zip files in the specified input directory
    zip_files = glob.glob(os.path.join(configuration["input_images_folder"], "*.zip"))

    for zip_file in zip_files:
        name = os.path.basename(zip_file)[4:-4]  # Extract the base name for the output files
        configuration["input_zip"] = zip_file  # Update configuration with the current zip file

        # Read and process cloud, shadow, and no data masks
        cloud_mask, shadow_mask, nodata_mask, dataSource = readCloudShadowMask(configuration, cloud)

        output_path_cloud = os.path.join(configuration["cloud_masks_path"], f"{name}_cloudMediumMask_Sen2Cor.tif")
        createGeoTifOneBand(output_path_cloud, cloud_mask, 0, dataSource, gdal.GDT_Byte)

        output_path_shadow = os.path.join(configuration["shadow_masks_path"], f"{name}_shadowMask_Sen2Cor.tif")
        createGeoTifOneBand(output_path_shadow, shadow_mask, 0, dataSource, gdal.GDT_Byte)

        #output_path_nodata = os.path.join(configuration["nodata_masks_path"], f"{name}_nodataMask_Sen2Cor.tif")
        #createGeoTifOneBand(output_path_nodata, nodata_mask, 0, dataSource, gdal.GDT_Byte)



def readCloudShadowMaskLandsat(configuration, verbose = False):
    """
    Extracts and processes cloud, shadow and snow masks from a Landsat zip file.

    Parameters:
    - configuration: Application parameters.

    Steps:
    1. Open the Landsat zip file.
    2. List all files in the zip archive and identify the appropriate QA pixel file.
    3. Extract the identified file to the specified output directory.
    4. Open the extracted file with GDAL and read the raster data.
    5. Convert the QA pixel values to binary and create corresponding masks for clouds, shadows, snow and no data.
    6. Combine the binary masks into a final mask, identifying different classes (clouds, shadows, snow, etc.).
    7. Return the cloud, shadow, snow and no data masks along with the GDAL dataset object.
    8. Clean up by deleting the extracted file after processing.
    """
    logging.info("Reading Landsat .zip data")

    with zipfile.ZipFile(configuration["input_zip"], 'r') as zip_ref:

        # List all files in the zip
        flist = [(s, s.split("/")[-1]) for s in zip_ref.namelist()]
        chn_fn = None

        if verbose:
            logging.info_list(flist, f"Zip content:")    

        # Find the relevant .tif file for the QA pixel data
        for (archive_name, filename) in flist:
            if archive_name.find('_QA_PIXEL') > -1 and (archive_name.endswith('.TIF') or archive_name.endswith('.tif')):
                chn_fn = (archive_name, filename)
                break
        
        if chn_fn is None:
            raise ValueError('Cannot find cloud channel name in zip file: {b}'.format(b=configuration["input_zip"]))

        # Extract the file to the specified output directory
        logging.info(f" -> Found right file: {chn_fn[0]}")
        extracted_path = os.path.join(configuration["output_folder_path"], chn_fn[1])
        with zip_ref.open(chn_fn[0]) as source, open(extracted_path, 'wb') as target:
            shutil.copyfileobj(source, target)

        # Ensure the file was extracted successfully
        if not os.path.exists(extracted_path):
            raise RuntimeError(f"File not found after extraction: {extracted_path}")

        # Open the extracted file with GDAL
        logging.info(f" -> Verified path for GDAL: {extracted_path}")
        ds = gdal.Open(extracted_path, 0)
        band_ds = ds.GetRasterBand(1)
        img_qa = (band_ds.ReadAsArray()).astype(np.uint16)

        # Analyze the QA pixel values to generate corresponding masks
        img_qa_values = np.unique(img_qa)  # Unique QA values
        mask_qa_values = np.zeros(len(img_qa_values), dtype=np.uint8)  # Initialize mask values
        binary_qa_values = []  # To store binary representations of QA values
        
        for i in range(len(img_qa_values)):
            b = np.binary_repr(img_qa_values[i], width=16)  # Convert to binary
            binary_qa_values.append(b)
            
            # SNOW: Check specific bits in the binary representation
            if b[-6] == '1':
                mask_qa_values[i] = 5
            
            # CLOUD SHADOWS: Check specific bits in the binary representation
            if b[-5] == '1' or (b[-11] == '1' and b[-12] == '1'):
                mask_qa_values[i] = 4
            
            # CLOUDS: medium-prob and high-prob cloud and cirrus
            if (b[-3] == '1' or b[-4] == '1' or (b[-9] == '0' and b[-10] == '1') or (b[-9] == '1' and b[-10] == '1') or (b[-15] == '1' and b[-16] == '1')):          
                mask_qa_values[i] = 3
                
            # FILL (NAN): Check the fill bit
            if b[-1] == '1':
                mask_qa_values[i] = 2

        # Create the final mask based on the QA values
        height, width = img_qa.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(len(img_qa_values)):
            layer = (img_qa == img_qa_values[i])
            mask[layer] = mask_qa_values[i]
            
        mask = mask.astype(int)
        shadow_mask = mask == 4  # Shadow mask
        cloud_mask = mask == 3  # Cloud mask
        snow_mask = mask == 5  # Snow mask
        nodata_mask = mask == 2  # No data mask

    logging.info(f" -> File {chn_fn[1]} extracted at: {extracted_path}")
    os.remove(extracted_path) 
    return cloud_mask, shadow_mask, nodata_mask, snow_mask, ds 

def unzipLandsatandSaveToTif(configuration):
    """
    Extracts cloud, shadow, snow, and no data masks from a Landsat zip file and saves them as GeoTIFFs.

    Parameters:
    - configuration: Application parameters.

    Steps:
    1. Read and process the cloud, shadow, snow and no data masks from the zip file.
    2. If the area is "Siberia", create and save the snow mask as a GeoTIFF in the specified directory.
    3. Create and save the cloud mask as a GeoTIFF in the specified directory.
    4. Create and save the shadow mask as a GeoTIFF in the specified directory.
    5. Create and save the no data mask as a GeoTIFF in the specified directory.
    """
    name = os.path.basename(configuration["input_zip"])[:-4]  # Extract the base name for the output files

    # Read and process cloud, shadow, snow, and no data masks
    cloud_mask, shadow_mask, nodata_mask, snow_mask, dataSorce = readCloudShadowMaskLandsat(configuration)

    #if configuration["area"].lower() == "siberia":
       # snow_mask = readSnowMaskLandsat(zip_path, snow)
        #output_path_snow = os.path.join(configuration["snow_masks_path"], name + '_snowIceMask.tif')
        #logging.info(f" -> Creating .TIF for snow mask at: {output_path_snow}")
        #createGeoTifOneBand(output_path_snow, snow_mask, 0, dataSorce, gdal.GDT_Byte)

    output_path_cloud = os.path.join(configuration["cloud_masks_path"], name + '_cloudMediumMask.tif')
    logging.info(f" -> Creating .TIF for cloud mask at: {output_path_cloud}")
    createGeoTifOneBand(output_path_cloud, cloud_mask, 0, dataSorce, gdal.GDT_Byte)
    
    output_path_shadow = os.path.join(configuration["shadow_masks_path"], name + '_shadowMask.tif')
    logging.info(f" -> Creating .TIF for shadow mask at: {output_path_shadow}")
    createGeoTifOneBand(output_path_shadow, shadow_mask, 0, dataSorce, gdal.GDT_Byte)
    
    #output_path_nodata = os.path.join(configuration["nodata_masks_path"], name + '_nodataMask.tif')
    #logging.info(f" -> Creating .TIF for no data mask at: {output_path_nodata}")
    #createGeoTifOneBand(output_path_nodata, nodata_mask, 0, dataSorce, gdal.GDT_Byte)



###################################################
#   STAGE 2 - Create seasonal background images   #
###################################################

def loadTimeSeries2D(n_imgs, pix_region, patch_size, cloudMasks, shadowMasks):
    """
    Loads a 2D time series of image patches, applying cloud and shadow masks.

    Parameters:
    - n_imgs: List of image file paths to be loaded as a time series.
    - pix_region: A 2D tuple (x, y) representing the top-left pixel of the patch.
    - patch_size: The size of the square patch to be extracted from each image.
    - cloudMasks: List of file paths to corresponding cloud masks.
    - shadowMasks: List of file paths to corresponding shadow masks.

    Steps:
    1. Initialize an empty 2D array to hold the time series data.
    2. For each image:
       a. Load the corresponding cloud and shadow masks and combine them.
       b. Load the image patch and apply the combined mask to remove cloud and shadow pixels.
       c. Flatten the patch to 1D and add it to the time series array.
    3. Clear any references to loaded datasets to free memory.
    4. Return the time series 2D array.
    """
    
    ts_2D = np.zeros((patch_size * patch_size, len(n_imgs)))  # Initialize the time series 2D array

    for i in range(len(n_imgs)):   

        # Load and process the cloud mask
        cmask_ds = gdal.Open(cloudMasks[i])
        cmask = cmask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)   
        
        # Load and process the shadow mask
        smask_ds = gdal.Open(shadowMasks[i])
        smask = smask_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
        
        # Combine cloud and shadow masks
        cloud_shadow_mask = (cmask + smask)
        not_cloud_shadow_mask = cloud_shadow_mask ^ 1  # Invert the mask

        with zipfile.ZipFile(n_imgs[i], 'r') as zip_ref:

            season_img_unzip = readSentinelZipImage(zip_ref, n_imgs[i], configuration, ['B02'], 10)['B02']

            # Load the image and apply the mask
            img_temp_ds = gdal.Open(season_img_unzip)
            img_temp = np.zeros((patch_size, patch_size))
            img_temp[:, :] = img_temp_ds.GetRasterBand(1).ReadAsArray(int(pix_region[0]), int(pix_region[1]), patch_size, patch_size)
            
        img_temp[:, :] = np.multiply(img_temp[:, :], not_cloud_shadow_mask)
        
        # Replace masked-out pixels with NaN and flatten the array
        img_temp[img_temp == 0] = ['nan']       
        img_temp = np.reshape(img_temp, (patch_size * patch_size))
       
        ts_2D[:, i] = img_temp  # Add the processed patch to the time series array
            
    # Clear references to datasets to free memory
    cmask = None
    cmask_ds = None
    smask = None
    smask_ds = None
    img_temp = None
    img_temp_ds = None
    
    return ts_2D  # Return the 2D time series array


def getBackgroundImage(configuration, verbose = False):
    """
    Generates a seasonal background image by processing time series data from Sentinel-2 images.

    Parameters:
    - configuration: Application parameters.

    Steps:
    1. Load and sort all cloud masks, shadow masks and images from the specified directories.
    2. Define the seasons (groups of months) to process separately.
    3. For each season:
       a. Filter the masks and images to include only those matching the season and year.
       b. Initialize an empty array to store the seasonal background image.
       c. Process the image in patches:
           i. Load the time series data for each patch, applying masks.
           ii. Compute the quantile value for each patch and store it in the background image.
       d. Save the seasonal background image as a GeoTIFF.
    """
    
    # Load and sort cloud masks, shadow masks, and images
    tile = configuration["tile"]
    cloudMasks_all = sorted(glob.glob(configuration["cloud_masks_path"] + f'/*{tile}*cloudMediumMask_Sen2Cor.tif'))   
    shadowMasks_all = sorted(glob.glob(configuration["shadow_masks_path"] + f'/*{tile}*shadowMask_Sen2Cor.tif'))   
    img_all = sorted(glob.glob(configuration["input_images_folder"] + f'/*{tile}*.zip'))  
    year = str(configuration['year'])

    if verbose:
        logging.info_list(img_all, f"Founded {len(img_all)} images:")
        logging.info_list(cloudMasks_all, f"Founded {len(cloudMasks_all)} Cloud masks:")
        logging.info_list(shadowMasks_all, f"Founded {len(shadowMasks_all)} Shadow masks:")

    # Define seasons (months grouped into four seasons)
    seasons = []
    seasons.append(['01', '02', '03'])  # January, February, March  
    seasons.append(['04', '05', '06'])  # April, May, June
    seasons.append(['07', '08', '09'])  # July, August, September
    seasons.append(['10', '11', '12'])  # October, November, December

    # Process each season
    logging.info("Processing seasons:")
    
    for s in range(len(seasons)):
        season = seasons[s]
        logging.info(f"Generating background for season {s} ({season})")

        # Filter masks and images for the current season
        cloudMasks = sorted([x for x in cloudMasks_all if year + season[0] in x or year + season[1] in x or year + season[2] in x ])
        shadowMasks = sorted([x for x in shadowMasks_all if year + season[0] in x or year + season[1] in x or year + season[2] in x ])
        season_img = sorted([x for x in img_all if year + season[0] in x or year + season[1] in x or year + season[2] in x ])
        
        logging.info(f"Number of elements for the current season: {len(shadowMasks)}")
        # Ensure the number of masks matches the number of images
        if len(shadowMasks) != len(cloudMasks) or len(season_img) != len(cloudMasks):
            raise IOError("The number of clouds and shadow masks should correspond to the number of Sentinel images!")
       
        if len(shadowMasks) != 0:
            with zipfile.ZipFile(season_img[0], 'r') as zip_ref:

                season_img_unzip = readSentinelZipImage(zip_ref, season_img[0], configuration, ['B02'], 10)['B02']
                target_ds = gdal.Open(season_img_unzip)
                nl = int(target_ds.RasterYSize)
                nc = int(target_ds.RasterXSize)
                
                sesonalBackground = np.zeros((nl, nc))  # Initialize the seasonal background array
                patch_sizeX = 2196 
                patch_sizeY = 2196 
                
                # Process the image in patches
                for i in tqdm(range(5), desc="Processing patches in X direction", leave=False):
                    for j in tqdm(range(5), desc="Processing patches in Y direction", leave=False):
                        pix_region = np.array([i * patch_sizeX, j * patch_sizeX])
                        img3d = loadTimeSeries2D(season_img, pix_region, patch_sizeX, cloudMasks, shadowMasks)
                        
                        # Compute the quantile value for each patch
                        arr = img3d
                        mask = (arr >= np.nanmin(arr)).astype(int)
                        count = mask.sum(axis=1)
                        groups = np.unique(count)
                        groups = groups[groups > 0]
                        
                        temp = np.zeros((arr.shape[0]))
                        for g in range(len(groups)):
                            pos = np.where(count == groups[g])
                            values = arr[pos]
                            values = np.nan_to_num(values, 0)  # Replace NaNs with 0
                            values = np.sort(values, axis=1)
                            values = values[:, -groups[g]:]
                            temp[pos] = np.percentile(values, 25, axis=1, method='midpoint')
                            
                        sesonalBackground[j * patch_sizeX:j * patch_sizeX + patch_sizeX, i * patch_sizeX:i * patch_sizeX + patch_sizeX] = np.reshape(temp, [patch_sizeX, patch_sizeY])
                    
                    
##            cloud_shadow_mask = []
##            for i in range(len(season_img)):
##                cmask_ds = gdal.Open(cloudMasks[i])
##                cmask = cmask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
##                
##                smask_ds = gdal.Open(shadowMasks[i])
##                smask = smask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
##                cloud_shadow_mask.append(cmask + smask)
##            cloud_shadow_mask = np.prod(cloud_shadow_mask, axis=0, dtype=bool)
##            
##            sesonalBackground[cloud_shadow_mask] = ['nan']
##            sesonalBackground = np.nan_to_num(sesonalBackground, nan=np.nanmin(sesonalBackground))

                # Save the seasonal background image as a GeoTIFF
                background_output_path = os.path.join(str(configuration["seasonal_bg_path"]), 'backgroundImage_' + configuration["tile"] + year + '_' + str(s + 1) + '.tif')
                createGeoTifOneBand(background_output_path, sesonalBackground, 0, target_ds, gdal.GDT_Float32)
                target_ds = None  # Clear the dataset reference

                del target_ds  # Delete the dataset object to free memory


#######################################################################
#   STAGE 3 - Refine Cloud and Shadow Masks Using Background Images   #
#######################################################################

def refineCloudMask(configuration, input_cloud_mask_path, background_path, refined_cloud_mask_path):
    """
    Refines a cloud mask using a seasonal background image and K-means clustering.

    Parameters:
    - configuration: Application parameters.
    - refined_cloud_mask_path: The output path where the final cloud mask will be saved.
    - input_path_cloud: The file path to the initial cloud mask.
    - background_path: The file path to the seasonal background image.

    Steps:
    1. Load the background image and the current image.
    2. Load the initial cloud mask and calculate the mean cloud mask value.
    3. Calculate the absolute difference between the input image and the background image.
    4. Calculate the percentage of cloud cover in the initial cloud mask.
    5. If cloud cover is above a threshold, cluster the difference image into three clusters using K-means.
    6. Identify the cluster that best matches the mean cloud mask value and combine it with the initial mask.
    7. Save the final refined cloud mask as a GeoTIFF file.
    """

    mean = np.zeros(3)  # Initialize an array to store mean values of the clusters
    mask = np.zeros(3)  # Initialize an array to store masks for each cluster
    dist = np.zeros(3)  # Initialize an array to store distance from mean cloud mask value
    
    # Load the seasonal background image (cloudless image for comparison)
    background_ds = gdal.Open(background_path)
    background_img = background_ds.GetRasterBand(1).ReadAsArray(0, 0, 10980, 10980)
    
    finalMask = np.zeros((10980, 10980))  # Initialize the final cloud mask array
    
    # Load the input image to be processed
    img_temp_ds = gdal.Open(configuration["input_bands"]["B02"])
    nl = int(img_temp_ds.RasterYSize)
    nc = int(img_temp_ds.RasterXSize)
    img_temp = img_temp_ds.GetRasterBand(1).ReadAsArray(0, 0, nl, nc)
    
    # Load the initial cloud mask generated by Sen2Cor or Fmask
    cloudMask_ds = gdal.Open(input_cloud_mask_path)
    cloudMask = cloudMask_ds.GetRasterBand(1).ReadAsArray(0, 0, nl, nc)
    
    # Calculate the mean value of pixels labeled as cloud in the initial cloud mask
    meanCloudMask = (cloudMask * img_temp).astype(float)
    meanCloudMask[meanCloudMask == 0] = ['nan']  # Replace zeros with NaN for accurate mean calculation
    meanCloudMask = np.nanmean(meanCloudMask)
    
    # Calculate the absolute difference between the input image and the background image
    diff = abs(img_temp - background_img)
    
    # Calculate the percentage of cloud cover in the initial cloud mask
    num = np.unique(cloudMask, return_counts=True)[1]
    cloudPerc = num[1] / (num[0] + num[1])
    
    # If cloud cover is above a threshold (0.08%), proceed with clustering
    if cloudPerc > configuration["cloud_coverage_threshold"]:
        # Reshape the difference image into a 1D array for clustering
        diff = diff.reshape(-1, 1)
        
        # Apply K-means clustering to group pixels into 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=0).fit(diff)
        cluster = np.reshape(kmeans.labels_, [10980, 10980])
        diff = diff.reshape(10980, 10980)
        
        # Analyze each cluster to find the one most similar to the mean cloud mask value
        for c in range(3):
            mask = ([cluster == c][0])
            masked = diff * mask
            masked[masked == 0] = ['nan']  # Replace zeros with NaN for accurate mean calculation
            mean[c] = np.nanmean(masked)
            dist[c] = np.abs(meanCloudMask - mean[c])  # Calculate distance from mean cloud mask value
        
        # Identify the cluster that is most similar to the mean cloud mask value
        ix = np.argsort(dist)
        mask = ([cluster == ix[0]][0])
        finalMask = mask.astype(int) + cloudMask  # Combine the selected cluster with the initial cloud mask
    else:
        finalMask = cloudMask  # If cloud cover is too low, use the initial cloud mask as the final mask
    
    # Ensure the final mask contains only binary values (0 or 1)
    finalMask[finalMask > 1] = 1
    
    # Save the final refined cloud mask as a GeoTIFF file
    createGeoTifOneBand(refined_cloud_mask_path, finalMask, 0, cloudMask_ds, gdal.GDT_Byte)
    
    # Clean up by clearing the datasets from memory
    img_temp_ds = None
    cloudMask_ds = None
    del img_temp_ds
    del cloudMask_ds

    
       
def refineShadowdMask(configuration, input_shadow_mask_path, refined_cloud_mask_path, refined_shadow_mask_path):
    """
    Refines a shadow mask using spectral bands (Blue, NIR, SWIR) and the cloud mask.

    Parameters:
    - configuration: Application parameters.
    - input_shadow_mask_path: The file path to the initial shadow mask.
    - refined_cloud_mask_path: The file path to the final cloud mask.
    - refined_shadow_mask_path: The output path where the final shadow mask will be saved.

    Steps:
    1. Load the input image, final cloud mask, and initial shadow mask.
    2. Calculate the mean cloud mask value within the image.
    3. If cloud cover and mean value thresholds are met, proceed with shadow detection.
    4. Compute the Cloud-Shadow Index (CSI) and Blue band thresholds to identify shadow regions.
    5. Combine the binary shadow detection masks with the initial shadow mask.
    6. Apply a median filter to refine the shadow mask.
    7. Save the final refined shadow mask as a GeoTIFF file.
    """
    

    blue_band_path = configuration["input_bands"]['B02']
    nir_band_path = configuration["input_bands"]['B07']
    swir11_band_path = configuration["input_bands"]['B09']
    
    # Load the input image to be processed
    blue_image = gdal.Open(blue_band_path)
    nir_image = gdal.Open(nir_band_path)
    swir11_image  = gdal.Open(swir11_band_path)
    
    num_bands = blue_image.RasterCount
    #logging.info(f"The blue image has {num_bands} band(s).")
    #logging.info(f"The nir image has {nir_image.RasterCount} band(s).")
    #logging.info(f"The swir image has {swir11_image.RasterCount} band(s).")

    sizeX = int(blue_image.RasterYSize)
    finalShadowMask = np.zeros((sizeX, sizeX))  # Initialize the final shadow mask array
    
    # Load the relevant spectral bands (Blue, NIR, SWIR) from the input image
    img_patch = blue_image.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)
    
    # Load the final cloud mask
    cloudMask_ds = gdal.Open(refined_cloud_mask_path)
    cloud = cloudMask_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)
    
    # Load the initial shadow mask
    shadow_ds = gdal.Open(input_shadow_mask_path)
    shadow = shadow_ds.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)
    
    # Calculate the mean value of pixels labeled as cloud in the final cloud mask
    meanCloudMask_patch = (cloud * img_patch).astype(float)
    meanCloudMask_patch[meanCloudMask_patch == 0] = ['nan']  # Replace zeros with NaN for accurate mean calculation
    meanCloudMask_patch = np.nanmean(meanCloudMask_patch)
    
    # Calculate the percentage of cloud cover in the final cloud mask
    num = np.unique(cloud, return_counts=True)[1]
    cloudPerc = num[1] / (num[0] + num[1])
    

    # If cloud cover and mean cloud mask thresholds are met, proceed with shadow detection
    if cloudPerc > configuration["cloud_coverage_threshold"] and meanCloudMask_patch > configuration["image_brightness_threshold"]:

        # Load the Blue, NIR, and SWIR bands from the image
        blue = (blue_image.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)).astype(float)
        nir = (nir_image.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)).astype(float)
        swir11 = (swir11_image.GetRasterBand(1).ReadAsArray(0, 0, sizeX, sizeX)).astype(float)
        
        # Mask out cloud pixels by setting them to NaN
        blue[cloud == 1] = ['nan']
        nir[cloud == 1] = ['nan']
        swir11[cloud == 1] = ['nan']
        
        # Compute the Cloud-Shadow Index (CSI) by averaging NIR and SWIR bands
        csi = (nir + swir11) / 2
        
        # Calculate thresholds for CSI and Blue bands
        t1 = np.nanmin(csi) + 0.5 * (np.nanmean(csi) - np.nanmin(csi))
        t2 = np.nanmin(blue) + 0.25 * (np.nanmean(blue) - np.nanmin(blue))
        
        # Create binary masks by thresholding CSI and Blue bands
        csi_th = csi <= t1
        blue_th = blue <= t2
        
        # Combine the binary masks to detect shadow regions
        mask = (csi_th * blue_th).astype(np.uint8)
        
        # Refine the shadow mask using a median filter
        finalShadowMask = medfilt2d(mask, kernel_size=3)
        
        # Combine the refined shadow mask with the initial shadow mask
        finalShadowMask = finalShadowMask + shadow
    else:
        finalShadowMask = shadow  # If conditions are not met, use the initial shadow mask as the final mask
    
    # Ensure the final mask contains only binary values (0 or 1)
    finalShadowMask[finalShadowMask > 1] = 1
    
    # Save the final refined shadow mask as a GeoTIFF file
    createGeoTifOneBand(refined_shadow_mask_path, finalShadowMask, 0, cloudMask_ds, gdal.GDT_Byte)
    
    # Clean up by clearing the datasets from memory
    img_temp_ds = None
    cloudMask_ds = None
    shadow_ds = None
    del img_temp_ds
    del cloudMask_ds
    del shadow_ds
        
if __name__ == "__main__":   

    configuration = retrieve_configuration()

    # Parameters setup
    performance_file = setup_logger(configuration)
    
    # Start memory monitoring in a separate thread
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    gdal.PushErrorHandler(gdal_error_handler)
    gdal.AllRegister()
    gdal.UseExceptions() # Due to a warning, if gdal base image is update, this can may be removed.

    
    verbose = configuration["verbose"]

    configuration["seasonal_bg_path"] = configuration["output_folder_path"] + "/backgrounds"
    configuration["cloud_masks_path"] = configuration["output_folder_path"] + "/cloud_masks"
    configuration["shadow_masks_path"] = configuration["output_folder_path"] +"/shadow_masks"
    configuration["nodata_masks_path"] = configuration["output_folder_path"] +"/nodata_masks"
    configuration["snow_masks_path"] = configuration["output_folder_path"] + "/snow_masks"

    create_folder_if_not_exists(configuration["output_folder_path"])
    create_folder_if_not_exists(configuration["seasonal_bg_path"])
    create_folder_if_not_exists(configuration["cloud_masks_path"])
    create_folder_if_not_exists(configuration["shadow_masks_path"])
    #create_folder_if_not_exists(configuration["nodata_masks_path"])
    #create_folder_if_not_exists(configuration["snow_masks_path"])

    logging.info(f"Cloud and Shadow detection - Application parameters:")
    print_map(configuration)

    logging.info(f"-------- STAGE 1 --------")

    use_stage = configuration['use_stage'] == True
    
    if (use_stage and configuration["stage"] == 1) or (use_stage == False):

        # this step read Sen2cor masks. It must be runned once specifying the folder.
        # For each *.zip file the sen2cor masks are extracted.

        logging.info(f" - Reading Sen2Cor mask for {configuration['area']} in {configuration['year']}")

        if configuration["sensor"] == "S2": 
            unzipS2andSaveToTif(configuration)
        else:
            unzipLandsatandSaveToTif(configuration)      


    if (use_stage and configuration["stage"] == 2) or (use_stage == False): 

        # this condition should be run once for each S2 tile 
        # (after Sen2cor mask has been read)
        logging.info(f"-------- STAGE 2 --------")

        logging.info(f"Generating background image for {configuration['area']} in {configuration['year']}")
        getBackgroundImage(configuration)
    
    if (use_stage and configuration["stage"] == 3) or (use_stage == False): 
    
        logging.info(f"-------- STAGE 3 --------")

        # this condition should be run once for each S2 image 
        # (after Sen2cor mask and backgorund image have been read)    
        # generates cloud and shadow path correspodning to input_tif 
        img_all = sorted(glob.glob(configuration["input_images_folder"] + '/*.zip'))  

        for i, img_path in enumerate(img_all):
            logging.info(f"Refine cloud and shadow masks for {os.path.basename(img_path)}")
            with zipfile.ZipFile(img_path, 'r') as zip_ref:

                s2_image_bands =  readSentinelZipImage(zip_ref, img_path, configuration, bands = ["B02"], resolution=10, verbose=True)
                s2_image_bands.update(readSentinelZipImage(zip_ref, img_path, configuration, bands = ["B07"], resolution=20, verbose=True))
                s2_image_bands.update(readSentinelZipImage(zip_ref, img_path, configuration, bands = ["B09"], resolution=60, verbose=True))
                
                logging.info("Resampling B07 and B09 from 20m and 60m resolution to 10m.")
                s2_image_bands['B07'] = resample_band(gdal.Open(s2_image_bands['B07']), gdal.Open(s2_image_bands['B02']), 10)
                s2_image_bands['B09'] = resample_band(gdal.Open(s2_image_bands['B09']), gdal.Open(s2_image_bands['B02']), 10)

                img_name = os.path.basename(img_path)[4:-4]
                year = str(configuration["year"])
                sen2cor_cloud_mask_path = os.path.join(configuration["cloud_masks_path"], img_name + '_cloudMediumMask_Sen2Cor.tif')
                sen2cor_shadow_mask_path = os.path.join(configuration["shadow_masks_path"], img_name + '_shadowMask_Sen2Cor.tif')
                refined_cloud_mask_path =  os.path.join(configuration["cloud_masks_path"], img_name + '_cloudMediumMask.tif')
                refined_shadow_mask_path =  os.path.join(configuration["shadow_masks_path"], img_name + '_shadowMask.tif')
                    
                # Generates corresponding background image path
                img_month = img_name[img_name.index(year) + 4:img_name.index(year)+6]

                season = getSeason(img_month)
                background_path = os.path.join(configuration["seasonal_bg_path"], 'backgroundImage_' + configuration["tile"] + year + '_' + str(season) + '.tif')

                verbose = False
                if verbose:
                    logging.info("Paths & parameters:\n")
                    print(f" - Image name: {img_name}")
                    print(f" - S2 path cloud: {sen2cor_cloud_mask_path}")
                    print(f" - S2 path shadow: {sen2cor_shadow_mask_path}")
                    print(f" - Final cloud mask path: {refined_cloud_mask_path}")
                    print(f" - Final shadow mask path: {refined_shadow_mask_path}")
                    print(f" - Image month: {img_month}")
                    print(f" - Season: {season}")
                    print(f" - Background path: {background_path}")
                    
                logging.info("Refining cloud mask ...")
                configuration["input_bands"] = s2_image_bands
                refineCloudMask(configuration, sen2cor_cloud_mask_path, background_path, refined_cloud_mask_path )

                logging.info("Refining shadow mask ...")
                refineShadowdMask(configuration, sen2cor_shadow_mask_path, refined_cloud_mask_path, refined_shadow_mask_path)

    else:
       raise RuntimeError(f"Stage value specified is not correct. Allowed values are: 1,2,3. Specified: {configuration['stage']}")

    logging.info("Processing complete")


"""

# NOT USED - COMMENTED IN unzipLandsatAndSaveToTif
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
"""
