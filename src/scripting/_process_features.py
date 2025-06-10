import glob
import logging
import os
import xarray as xr

def load_features(path: str) -> dict:
    """
    Loads feature data from a specified path containing `.tif` files and 
    organizes them into dataset with features as variables.

    Parameters
    ----------
    path : str
        Path or pattern to the directory containing the feature `.tif` files. 
        Wildcards can be used to match multiple files.

    Returns
    -------
    dict
        A dictionary where keys are feature names (derived from the file names)
        and values are xarray DataArrays.

    Notes
    -----
    - Each `.tif` file is assumed to be a 2D image.
    - Feature names are extracted from the file names by splitting on underscores (`_`)
      and taking the first part.
    - All files for a given feature are concatenated along the "time" dimension to
      create a continuous temporal dataset. This is done to support a potential situation in which there are
      features calculated at multiple time steps in the specified folder. Actually not the case.
 
    """

    logger = logging.getLogger("load-features")

    # Initialize a dictionary to hold lists of DataArrays for each feature
    feature_data = {}

    # List all .tif files in the monthly directory
    tif_files = glob.glob(path)

    for tif_file in tif_files:
        feature_name = os.path.basename(tif_file).split('_')[0]

        # Load the feature .tif as a DataArray
        feature_da = xr.open_dataarray(
            tif_file,
            chunks="auto",
            engine="rasterio"
        ).squeeze()

        if feature_name not in feature_data:
            feature_data[feature_name] = [feature_da]
        else:
            feature_data[feature_name].append(feature_da)

    feature_datasets = {
        feature_name: xr.concat(data_arrays, dim='time')
        for feature_name, data_arrays in feature_data.items()
    }

    logger.info("Features loaded and organized into datasets.")
    return feature_datasets






