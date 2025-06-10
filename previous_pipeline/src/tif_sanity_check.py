import os
from osgeo import gdal
import sys

def verify_tif_integrity(tif_path):
    """
    Verify the integrity of a .tif file by attempting to open it with GDAL.

    :param tif_path: Path to the .tif file
    :return: True if the file is valid, False otherwise
    """
    try:
        gdal_file = gdal.Open(tif_path)
        if gdal_file is None:
            print(f"File could not be opened: {tif_path}")
            return False
        # Check if the first raster band can be accessed
        band = gdal_file.GetRasterBand(1)
        if band is None:
            print(f"Could not get raster band: {tif_path}")
            return False
        # Attempt to read data from the first band
        data = band.ReadAsArray()
        if data is None:
            print(f"Could not read data from raster band: {tif_path}")
            return False
        print(f"File is valid: {tif_path}")
        return True
    except Exception as e:
        print(f"Error reading file {tif_path}: {e}")
        return False

def verify_tif_files_in_directory(directory_path):
    """
    Verify the integrity of all .tif files in a given directory.

    :param directory_path: Path to the directory containing .tif files
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(root, file)
                verify_tif_integrity(tif_path)

def main(input_path):
    """
    Main function to verify the integrity of .tif files based on the input path.

    :param input_path: Path to a folder or a .tif file
    """
    if os.path.isdir(input_path):
        print(f"Scanning directory: {input_path}")
        verify_tif_files_in_directory(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".tif"):
        print(f"Verifying file: {input_path}")
        verify_tif_integrity(input_path)
    else:
        print("Invalid input. Please provide a path to a folder or a .tif file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tif_sanity_check.py <path to a folder or a tif file>")
    else:
        input_path = sys.argv[1]
        main(input_path)
