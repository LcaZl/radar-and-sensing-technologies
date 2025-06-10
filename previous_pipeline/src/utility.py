import sys
from osgeo import gdal
import os
import shutil
from tabulate import tabulate
import yaml
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
import time
import logging
import datetime
import csv


#################################
#   GENERAL UTILITY FUNCTIONS   #
#################################

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
        
def setup_logger(configuration):
    # Define the base repository path
    log_path = f"{configuration['output_folder_path']}/logs"
    os.makedirs(log_path, exist_ok=True)

    # Find an incremental ID for the log file
    log_name = f"cloud-detection_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"
    log_file = os.path.join(log_path, log_name)
    performance_file = os.path.join(log_path, f"{log_name[:-4]}.csv")
    
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Optional: to see logs in the console
    logging.info(f"Logging to {log_file}")
    return performance_file



def monitor_memory(performance_file, interval=1):
    """
    Logs the memory usage every `interval` seconds and saves it to a CSV file.
    """
    process = psutil.Process()
    start_time = time.time()
    
    # Write the header for the CSV file
    with open(performance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Elapsed Time (s)", "Memory Usage (MB)"])
    
    while True:
        elapsed_time = time.time() - start_time
        memory_info = process.memory_info()
        memory_used_mb = memory_info.rss / (1024 ** 2)  # Resident Set Size in MB

        # Append memory usage data to the CSV file
        with open(performance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([elapsed_time, memory_used_mb])
        
        time.sleep(interval)
        
###########################################
#   PREDEFINED FILE SYSTEM INTERACTIONS   #
###########################################

def get_filename(path_to_file):
    """
    Extract the name of the .zip file from a given path.

    :param zip_path: Path to the .zip file
    :return: Name of the .zip file without extension
    """
    # Get the base name of the file
    file_name = os.path.basename(path_to_file)
    # Split the name and extension, return the name part
    name, _ = os.path.splitext(file_name)
    return name

def create_folder(folder_path):
    """
    Create a folder or delete and recreate it if it already exists.

    :param folder_path: Path to the folder to be created or recreated
    """
    if os.path.exists(folder_path):
        # If the folder exists, delete it and its contents
        shutil.rmtree(folder_path)
    # Create the folder
    os.makedirs(folder_path)

def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not already exist.

    :param folder_path: Path to the folder to be created
    """
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)

def retrieve_configuration():
    """
    Reads a YAML file from the given path and returns its contents as a dictionary.
    
    :param file_path: Path to the YAML file
    :return: Dictionary containing the YAML configuration
    """
    parser = argparse.ArgumentParser(description="Read a YAML configuration file.")
    parser.add_argument("config_file_path", type=str, help="Path to the YAML configuration file")
    
    # Parse and return the command-line arguments
    args = parser.parse_args()
    
    with open(args.config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

##############################
#   GDAL SUPPORT FUNCTIONS   #
##############################
def createGeoTif(configuration, array, dataSorce, dataType):
    """
    Creates a GeoTIFF file from a 3D numpy array.

    Parameters:
    - configuration: Application parameters.
    - array: The 3D numpy array to be saved.
    - dataSorce: A GDAL dataset object to copy geotransform and projection information from.
    - dataType: The data type of the output GeoTIFF.

    Steps:
    1. Get the GDAL driver for GeoTIFF.
    2. Replace NaNs in the array with the specified No Data Value.
    3. Create a new GeoTIFF file with the specified dimensions and data type.
    4. Set the geotransform and projection information from the source dataset.
    5. Write each band (layer) of the array to the GeoTIFF file.
    6. Save the file, free memory and return the file name.
    """
    name = configuration["tif_output_path"]
    driver = gdal.GetDriverByName('GTiff')
    array[np.isnan(array)] = configuration["no_data_value"]
    dataset = driver.Create(name, array.shape[1], array.shape[0], array.shape[2], dataType)
    dataset.SetGeoTransform(dataSorce.GetGeoTransform())
    dataset.SetProjection(dataSorce.GetProjection()) 

    for i in range(0, array.shape[2]):
        dataset.GetRasterBand(i+1).WriteArray(array[:, :, i])
    dataset.FlushCache()

    dataSet = None
    del dataSet
    return name

def createGeoTifOneBand(name, array, NDV, dataSorce, dataType):
    """
    Creates a single-band GeoTIFF file from a 2D numpy array.

    Parameters:
    - name: The output file name.
    - array: The 2D numpy array to be saved.
    - NDV: No Data value to replace NaNs in the array.
    - dataSorce: A GDAL dataset object to copy geotransform and projection information from.
    - dataType: The data type of the output GeoTIFF.

    Steps:
    1. Get the GDAL driver for GeoTIFF.
    2. Replace NaNs in the array with the specified No Data Value (NDV).
    3. Create a new GeoTIFF file with the specified dimensions and data type.
    4. Set the geotransform and projection information from the source dataset.
    5. Set the NDV value and write the array to the GeoTIFF file as a single band.
    6. Save the file, free memory and return the file name.
    """
    
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
    print(f"Created .TIF file: {name}")
    
    return name
 


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
    print('\tMessage: %s' % err_msg, file=sys.stderr)

def gdal_file_info(gdal_file):
    """
    Print information related to the input GDAL file.

    :param gdal_file: GDAL file to inspect.
    """
    print(f"\n----------------------------------------")
    print(f"|            GDAL FILE INFO            |")
    print(f"----------------------------------------")
    print(f"NAME: {os.path.basename(gdal_file.GetDescription())}\n")

    if gdal_file is None:
        print("GDAL could not open file.")
        return

    # Print driver information
    print(f"Driver: {gdal_file.GetDriver().ShortName}/{gdal_file.GetDriver().LongName}")

    # Print raster size
    print(f"Size is {gdal_file.RasterXSize} x {gdal_file.RasterYSize} x {gdal_file.RasterCount}")

    # Print projection
    print(f"Projection is {gdal_file.GetProjection()}")

    # Get and print geotransform
    geotransform = gdal_file.GetGeoTransform()
    if geotransform:
        print(f"Origin = ({geotransform[0]}, {geotransform[3]})")
        print(f"Pixel Size = ({geotransform[1]}, {geotransform[5]})")
        print(f"Rotation (skew) = ({geotransform[2]}, {geotransform[4]})")

    # Print metadata
    metadata = gdal_file.GetMetadata()
    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    # Print information about each band
    for band_number in range(1, gdal_file.RasterCount + 1):
        band = gdal_file.GetRasterBand(band_number)
        print(f"\nBand {band_number}:")
        print(f"  Data Type: {gdal.GetDataTypeName(band.DataType)}")
        print(f"  Size: {band.XSize} x {band.YSize}")
        print(f"  Min: {band.GetMinimum()}")
        print(f"  Max: {band.GetMaximum()}")
        print(f"  NoData Value: {band.GetNoDataValue()}")
        print(f"  Block Size: {band.GetBlockSize()[0]} x {band.GetBlockSize()[1]}")

        # Print band metadata
        band_metadata = band.GetMetadata()
        if band_metadata:
            print("  Metadata:")
            for key, value in band_metadata.items():
                print(f"    {key}: {value}")

        # Print band statistics
        stats = band.GetStatistics(True, True)
        if stats:
            print("  Statistics:")
            print(f"    Minimum: {stats[0]:.3f}")
            print(f"    Maximum: {stats[1]:.3f}")
            print(f"    Mean: {stats[2]:.3f}")
            print(f"    StdDev: {stats[3]:.3f}")

        # Print color interpretation
        color_interp = band.GetColorInterpretation()
        print(f"  Color Interpretation: {gdal.GetColorInterpretationName(color_interp)}")

        # Check if band has a color table
        color_table = band.GetColorTable()
        if color_table:
            print(f"  Color Table: {color_table.GetCount()} entries")


    print(f"----------------------------------------")
    print(f"|            END FILE INFO             |")
    print(f"----------------------------------------\n")


#####################################
#   OUTPUT PRESENTATION FUNCTIONS   #
#####################################

# Present in a structured way the following data structures: list, dataframe.

def print_list(lis, title = ""):
    print(title)
    if len(lis) == 0:
        print(" -- Empty list.")
    else:
        for i, el in enumerate(lis):
            print(f" - {i+1} - {el}")
            
def print_dataframe(df, title='', columns_to_exclude=[]):
    if title:
        print(title)
    if df is not None and len(df) != 0:
        df_to_show = df.drop(columns=columns_to_exclude)
        print(tabulate(df_to_show, headers='keys', tablefmt='psql'))
    else:
        print("No data to display.")

def print_map(config_map, title = None):
    if title:
        print(title)
        
    for i, (key, value) in enumerate(config_map.items(), 1):
        print(f"{i+1} - {key} : {value}")
    
    
def print_dict(d, title = '', avoid_keys = []):
    if title is not None:
        print(title)
    for key, value in d.items():
        if key not in avoid_keys:
                print(f" - {key}: {value}")
