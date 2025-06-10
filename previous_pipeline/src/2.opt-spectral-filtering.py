
"""
3.2 Spectral Filtering 

The spectral filtering aims to detect and remove the outlier present in the optical images. To this end, in this step 
we  discard  the  reflectance  values  higher  than  the  0.999  quantile  and  lower  than  the  0.001  quantile  of  each 
spectral  band.  All  the  images  considered  in  the  experiments  have  cloud  coverage  less  than  40%.  In  order  to 
mitigate  any  possible  effect  of  clouds  and  shadow  present  on  the  image,  they  have  been  detected  by  using 
Sen2cor mask and discarded from the quantitative evaluation.
"""

import zipfile
from PIL import Image
from scipy import ndimage
from osgeo import gdal

from utility import *

def help():
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

def getOverallMask(cloud_masks_file_path, shadow_masks_file_path):
    """
    Creates an overall mask combining cloud and shadow masks.

    Parameters:
    - cloud_masks_file_path: Path to the cloud mask file.
    - shadow_masks_file_path: Path to the shadow mask file.

    Steps:
    1. If both mask paths are None, return None.
    2. Open the cloud and shadow mask files using GDAL.
    3. Read the first raster band of each mask file and convert it to a boolean array.
    4. Sum the masks element-wise to create a combined mask.
    5. Return the combined mask.
    """
    if cloud_masks_file_path is None and shadow_masks_file_path is None:
        return None
    else:
        mask_path_list = [cloud_masks_file_path, shadow_masks_file_path]
        ds_list = [gdal.Open(path) for path in mask_path_list]
        mask_list = [ds.GetRasterBand(1).ReadAsArray().astype(bool) for ds in ds_list]
        return np.sum(mask_list, axis=0, dtype=bool)

def getCloudyStripesMask(dataset, save_path, cloud_shadow_mask):
    """
    Creates a mask for cloudy stripes in Landsat images.

    Parameters:
    - dataset: The GDAL dataset of the current band.
    - save_path: Path to save the new cloudy stripes mask file.
    - cloud_shadow_mask: The combined cloud and shadow mask.

    Steps:
    1. Read the first raster band of the dataset and convert it to a numpy array.
    2. Create a mask with the data equals the No Data Value.
    3. If a cloud-shadow mask is provided, dilate it and intersect with the stripes mask.
    4. Apply binary opening to remove small noise from the mask.
    5. Save the new mask to a GeoTIFF file.
    6. Return the cloudy stripes mask.

    Note:
    One could simply dilate enough the cloud-shadow mask in order to find the cloudy parts
    of stripes, and then erode (i.e., closing). However the dilation factor would be too big and worsen
    the cloud-shadow masks. Instead, explicitly estimating the cloudy stripes allow to use less dilation
    (or no dilation) on the cloud-shadows mask later in the function 'fillStripesLandsat'.
    """

    band_ds = dataset.GetRasterBand(1)

    data = (band_ds.ReadAsArray()).astype(np.uint16)

    # Compute the stripes mask from no data areas of the image
    stripes_mask = data==band_ds.GetNoDataValue()

    cloudy_stripes_mask = stripes_mask
    if cloud_shadow_mask is not None:
        # Dilate the cloud and shadow mask to connect masked areas split by the stripes
        dil_cloud_shadow_mask = ndimage.binary_closing(cloud_shadow_mask, iterations=10)

        # obtain only the cloudy parts of the stripes
        cloudy_stripes_mask = stripes_mask*dil_cloud_shadow_mask

    # delete small lines that are likely to be wrong
    cloudy_stripes_mask = ndimage.binary_opening(cloudy_stripes_mask, iterations=1)

    # Save the new mask to drive
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.CreateCopy(save_path,dataset,strict=0)
    new_band_ds = new_ds.GetRasterBand(1)
    new_band_ds.WriteArray(cloudy_stripes_mask)
    
    return cloudy_stripes_mask

def fillStripesLandsat(dataset, img_path, mask):
    """
    Fills in missing stripes in Landsat images, particularly for Landsat 7.

    Parameters:
    - dataset: The GDAL dataset of the current band.
    - img_path: Path for saving the filled image.
    - mask: The combined cloud and shadow mask.

    Steps:
    1. Create a copy of the dataset with a new name indicating stripes-filled.
    2. Use GDAL's FillNodata function to fill in the missing stripes.
    3. Apply binary closing to the mask to connect small gaps.
    4. Create a keep mask by inverting the combined mask.
    5. Apply the keep mask to the filled data and save it.
    6. Return the filled data array.
    """
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.CreateCopy(img_path + '_stripes-filled.TIF',dataset,strict=0)
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
    
def readSentinelData(configuration, bands):
    """
    Extracts and processes Sentinel-2 data from a zip file.

    Parameters:
    - configuration: Application parameters.
    - bands: List of band identifiers to be processed.

    Steps:
    1. Initialize a 3D numpy array to store the output image.
    2. Create a temporary directory for extraction.
    3. Open the zip file and list all its contents.
    4. For each specified band:
       a. Determine the resolution based on the band.
       b. Find the corresponding jp2 file in the zip archive.
       c. Extract the file to the temporary directory.
       d. Open the extracted file with GDAL.
       e. Read the raster band data.
       f. Resample the data if necessary to a resolution of 10m.
       g. Apply quantile-based filtering to remove outliers.
       h. Store the processed data in the output image array.
    5. Return the output image array and the GDAL dataset.
    """
    output_img = np.zeros((10980, 10980, len(bands)), dtype=np.uint16)
    dataSorce = None


    # Open the zip file
    with zipfile.ZipFile(configuration["zip_path"], 'r') as zip_ref:

        # List all files in the zip
        flist = [(s, s.split("/")[-1]) for s in zip_ref.namelist()]
        if VERBOSE:
            print_list(flist, f"\nZip content ({os.path.basename(configuration['zip_path'])}):")

        for i in range(len(bands)):
            band = bands[i]
            chn_fn = None
            print(f"\nExamining Band: {band}")

            if band in ['B02', 'B03', 'B04', 'B08']:
                res = 10
            elif band in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']:
                res = 20
            elif band in ['B01', 'B09']:
                res = 60
            else:
                raise ValueError(f'Bad band values: {band}')

            print(f" - Resolution for Band {band} is {res}m")

            # Check if the jp2 file is in the zip
            for (archive_name, filename) in flist:
                if archive_name.find(band) > -1 and archive_name.find('IMG_DATA') > -1 and archive_name.endswith('.jp2') and archive_name.find(f'{res}m') > -1 and archive_name.find(f'_T{configuration["tile"]}') > -1:
                    
                    chn_fn = (archive_name, filename)
                    break

            if chn_fn is None:
                raise ValueError(f'Cannot find channel name in zip file: {band}, {res}, tile={configuration["tile"]}')

            # Extract the file to the temporary directory

            print(f" -> Found right file: {chn_fn[0]}")
            extracted_path = os.path.join(configuration["output_folder_path"], chn_fn[1])
            with zip_ref.open(chn_fn[0]) as source, open(extracted_path, 'wb') as target:
                shutil.copyfileobj(source, target)

            print(f" -> Extracted file to output folder, verifying path ...")
            if not os.path.exists(extracted_path):
                raise RuntimeError(f"File not found after extraction: {extracted_path}")

            # Open the file using GDAL
            print(f" -> Verified path for GDAL: {extracted_path}")
            gdal_file = gdal.Open(extracted_path)
            if gdal_file is None:
                raise RuntimeError(f"GDAL could not open file: {extracted_path}")
            else:
                print(f" -> GDAL file successfully read.")

            if VERBOSE:
                gdal_file_info(gdal_file)

            band_ds = gdal_file.GetRasterBand(1)
            print(f" -> Raster band retrieved: {band_ds}")

            if band_ds is None:
                raise RuntimeError(f"Could not get raster band from file: {extracted_path}")

            data = (band_ds.ReadAsArray()).astype(np.uint16)

            if band == 'B02':
                dataSorce = gdal_file

            if res != 10:
                data = np.array(Image.fromarray(data).resize((10980, 10980), Image.NEAREST))

            band = np.reshape(data, (10980 * 10980, 1))
            maxVal = np.quantile(band, configuration["max_quantile_val"])
            minVal = np.quantile(band, configuration["min_quantile_val"])
            band[band > maxVal] = maxVal
            band[band < minVal] = minVal
            data = np.reshape(band, (10980, 10980))
            output_img[:, :, i] = data

    return output_img, dataSorce



def readLandsatData(configuration, bands):

    """
    Extracts and processes Landsat data from a zip file.

    Parameters:
    - configuration: Application parameters.
    - bands: List of band identifiers to be processed.

    Steps:
    1. Initialize an empty list to store the output image.
    2. Create a folder for output if it doesn't exist.
    3. Create a temporary directory for extraction.
    4. Open the zip file and list all its contents.
    5. For each specified band:
       a. Find the corresponding tif file in the zip archive.
       b. Extract the file to the temporary directory.
       c. Open the extracted file with GDAL.
       d. Read the raster band data.
       e. If the sensor is Landsat 7, handle missing stripes by filling them.
       f. Apply quantile-based filtering to remove outliers.
       g. Store the processed data in the output image list.
    6. Convert the list to a numpy array and move the axis.
    7. Return the output image array and the GDAL dataset.
    """
    
    output_img = []

    # Create a temporary directory
    # Open the zip file
    with zipfile.ZipFile(configuration["zip_path"], 'r') as zip_ref:

        # List all files in the zip
        zip_contents = [(s, s.split("/")[-1]) for s in zip_ref.namelist() if s.endswith(".TIF")]
        print_list([el[1] for el in zip_contents], f" - Zip content:")
        print(f" - Starting band examination ({len(bands)} bands to inspect)")

        for i in range(len(bands)):
            band = bands[i]
            chn_fn = None
            print(f"\n\nBAND: {band}")

            # Check if the tif file is in the zip
            for (archive_name, filename) in zip_contents:

                if VERBOSE: print(f" - Checking File: {filename}")

                if filename.find(band) > -1 and filename.endswith('.TIF') and 'T1' in filename:  # L2A
                    chn_fn = (archive_name, filename)
                    break

            if chn_fn is None:
                raise ValueError('Cannot find channel name in zip file: {b}'.format(b=band))

            print(f" -> Found right file: {chn_fn[0]}")
            extracted_path = os.path.join(configuration["output_folder_path"], chn_fn[1])
            with zip_ref.open(chn_fn[0]) as source, open(extracted_path, 'wb') as target:
                shutil.copyfileobj(source, target)

            print(f" -> Extracted file to output folder, verifying path ...")
            if not os.path.exists(extracted_path):
                raise RuntimeError(f"File not found after extraction: {extracted_path}")

            # Open the file using GDAL
            print(f" -> Verified path for GDAL: {extracted_path}")
            gdal_file = gdal.Open(extracted_path)
            if gdal_file is None:
                raise RuntimeError(f"GDAL could not open file: {extracted_path}")
            else:
                print(f" -> GDAL file successfully read.")

            if VERBOSE:
                gdal_file_info(gdal_file)

            band_ds = gdal_file.GetRasterBand(1)
            if band_ds is None:
                raise RuntimeError(f"Could not get raster band from file: {extracted_path}")

            data = (band_ds.ReadAsArray()).astype(np.uint16)

            nl = data.shape[0]
            nc = data.shape[1]
            print(f" -> Raster band retrieved (nl = {nl}, nc = {nc}): {band_ds}")

            if 'L7' in configuration["sensor"]:  # fill missing stripes Landsat 7 without reconstructing clouds and shadows
                # Do only once at the beginning of the loop

                if i == 0:
                    # Executed only one time
                    # read cloud and shadow masks for later filling missing stripes of Landsat 7
                    
                    print(" -> Reading cloud and shadows masks for Landsat 7 (ONCE).")
                    cloud_shadow_mask = getOverallMask(configuration["cloud_mask_path"], configuration["shadow_mask_path"])

                    # TODO: define a better save_path for cloudy_stripes_mask -> 
                    # -> Created a folder at the same level of the input .zip data and with the same name (not zipped)
                    cloudy_stripes_mask = getCloudyStripesMask(gdal_file, configuration["output_folder_path"] + "/cloudy_stripes_mask.TIF", cloud_shadow_mask)

                    print(f"     - Cloud mask: {type(cloud_shadow_mask)}")
                    print(f"     - Cloudt stripes mask: {type(cloudy_stripes_mask)}")

                    if cloud_shadow_mask  is None:
                        overall_mask = cloudy_stripes_mask
                    else:
                        overall_mask = cloud_shadow_mask + cloudy_stripes_mask

                print(" -> Filling stripes in Landsat 7 (ONCE).")
                data = fillStripesLandsat(gdal_file, configuration["output_folder_path"] + "/" + chn_fn[1], overall_mask)

            band = data[data != band_ds.GetNoDataValue()]
            maxVal = np.quantile(band, configuration["max_quantile_val"])
            minVal = np.quantile(band, configuration["min_quantile_val"])                
            data[(data > maxVal) * (data != band_ds.GetNoDataValue())] = maxVal                
            data[(data < minVal) * (data != band_ds.GetNoDataValue())] = minVal                
            output_img.append(data)

    output_img = np.array(output_img).astype(np.uint16)
    output_img = np.moveaxis(output_img, 0, -1)

    return output_img, gdal_file


if __name__ == "__main__":

    gdal.PushErrorHandler(gdal_error_handler)
    gdal.AllRegister()
    gdal.UseExceptions() # Due to a warning, if gdal base image is update, this can may be removed.
    

    # Read application parameters at the specified path.
    configuration = retrieve_configuration()
    configuration["zip_path"] = configuration["zip_path"] + "/" + configuration["file_name"] + ".zip"
    configuration["output_folder_path"] = configuration["output_folder_path"] + "/" + configuration["file_name"]
    configuration["tif_output_path"] = configuration["tif_output_path"] + "/" + configuration["file_name"] + ".tif"

    VERBOSE = configuration["verbose"]
    
    # Print the parameters
    print(f"\nSpectral Filtering\nApplication parameters:")
    print_map(configuration)
    print(f"\n{'Sentinel 2' if configuration['sensor'] == 'S2' else 'Landsat'} sensor specified.")

    # This folder will contain all the output produced by readSentinelData(...) and readLandsatData(...) functions. 
    create_folder(configuration["output_folder_path"])

    if configuration["sensor"] == "S2":

        bands = ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12']
        print(f" - Bands: {bands}")

        output_img, dataSorce = readSentinelData(configuration, bands)
        
    else:

        if 'L8' in configuration["sensor"]:
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        else:
            bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']

        print(f" - Bands: {bands}")

        output_img, dataSorce = readLandsatData(configuration, bands)

    # The final .tif file is stored in the same directory of the input .zip file.
    createGeoTif(configuration, output_img, dataSorce, gdal.GDT_UInt16)
    print("\nProcessing complete")

