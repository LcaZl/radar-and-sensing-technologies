import os
import xarray as xr
import pathlib
import typing
import numpy as np
import glob
import logging 

def filter_files(files: typing.List[str], year, month, tile) -> typing.List[str]:
    """
    Filters a list of composites file paths based on the specified year, month and tile.

    Parameters
    ----------
    files : list[str]
        A list of file paths to filter.
    year : int | None
        The year to filter files by. If None, no filtering by year is performed.
    month : str | None
        The month to filter files by. If None, no filtering by month is performed.
    tile : str | None
        The Sentinel-2 tile ID to filter files by. If None, no filtering by tile is performed.

    Returns
    -------
    list[str]
        A list of file paths that match the specified filters.

    Notes
    -----
    The function assumes that composite file names contain tile information and dates in 
    specific formats. 
    """

    filtered_files = []
    
    for file in files:

        file_name = os.path.basename(file)
        file_tile = file_name.split('_')[1][1:-4]
        file_year = file_name.split('_')[1][-4:]
        file_month = file_name.split("_")[-1].split('.')[0]

        # Filter conditions
        matches_year = year is None or int(file_year) == int(year)
        matches_month = month is None or str(file_month) == str(month)
        matches_tile = tile is None or str(file_tile) == str(tile)

        # Add file if all conditions match
        if matches_year and matches_month and matches_tile:
            filtered_files.append(file)
            
    return filtered_files


def apply_preprocess(
    dataset: xr.Dataset,
    file: str,
    preprocess: typing.Callable[[xr.Dataset], xr.Dataset] | None = None
) -> xr.Dataset:
    """
    Applies an optional preprocessing function to a composites dataset and assigns
    metadata including time, file name and tile ID.

    Parameters
    ----------
    dataset : xr.Dataset
        The input composites dataset to preprocess.
    file : str
        The file name associated with the dataset, used to extract metadata.
    preprocess : typing.Callable[[xr.Dataset], xr.Dataset] | None, optional
        A preprocessing function to apply to the dataset. If None, no
        preprocessing is performed.

    Returns
    -------
    xr.Dataset
        The dataset with preprocessing applied (if specified) and additional
        metadata assigned as coordinates and attributes.

    Notes
    -----
    - Adds a "time" coordinate based on the file name date, representing the
      first day of the month.
    - Expands the dataset to ensure "time" is a dimension.
    - Assigns the file name and tile ID as metadata to the dataset.
    - The function assumes that composite file names contain tile and year information 
    in specific formats.
    """
    
    if preprocess:
        dataset = preprocess(dataset)

    file_name = os.path.basename(file)
    tile = file_name.split('_')[1][1:-4]
    year = file_name.split('_')[1][-4:]
    month = file_name.split("_")[-1].split('.')[0]
    if len(month) == 1:
        month = f"0{month}"

    # Convert year-month to a numpy datetime64 (first day of the month)
    # Add '-01' to represent the first day
    date_np = np.datetime64(f"{year}-{month}-01", "ns")
            
    # Assign coordinates and attributes
    dataset = dataset.assign_coords({"time": date_np})
    dataset = dataset.expand_dims("time")  # Ensure time is a dimension
    dataset = dataset.assign_coords({"file_name": ("time", [file_name])})
    dataset = dataset.assign_attrs({"tile": tile})

    return dataset
    
def load_composites(
    path: typing.Union[str, pathlib.Path, list[str]],
    year : str = None,
    tile : str = None,
    month : str = None,
    preprocess: typing.Callable[[xr.Dataset], xr.Dataset] | None = None
):
    """Loads composites data from a given path or list of paths, optionally
    applying a preprocessing function and filtering the paths based on a requested tile and year.

    Parameters
    ----------
    path : list[str  |  pathlib.Path] | str | pathlib.Path
        The file path(s) to the Sentinel-2 data. Can be a single path, a list
        of paths, or a path pattern with wildcards.
    year : int | None
        Optional filter parameter
    tile : str | None
        Optinal filter parameter
    preprocess : typing.Callable[[xr.Dataset], xr.Dataset] | None
        An optional function to preprocess the data after loading. This is only
        applicable when loading multiple files.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The loaded xarray dataset containing the composites requested.

    Raises
    ------
    ValueError
        If 'preprocess' is provided when reading a single file or if the
        'preprocess' argument is not applicable. Also if the filter leaves with no data to load.
        
    """   

    logger = logging.getLogger("load-composites")
    
    if isinstance(path, str) and ("*" in path or "?" in path):
        path = glob.glob(str(path))  # Expand wildcard

    path = filter_files(path, year, month, tile)

    if len(path) == 0:
        raise ValueError(f"No composites found for tile {tile} and year {year}")
    
    if isinstance(path, list):
        
        ds = xr.open_mfdataset(
            path,
            chunks="auto",#{"time": -1, "band": 1, "y": 1098, "x": 10980},
            engine="rasterio",
            preprocess=lambda ds: apply_preprocess(
                dataset = ds, 
                file = os.path.basename(ds.encoding["source"]), 
                preprocess = preprocess),
            parallel=True,
            combine="by_coords",
        )
    
    else:
        
        # Single file case
        ds = xr.open_dataset(
            path,
            chunks="auto",
            engine="rasterio"
        )
        ds = apply_preprocess(ds, path)

    logger.info(f"Loaded {len(path)} composites for tile {tile} and year {year}.")

    return ds

