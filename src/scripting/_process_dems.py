import pathlib
import numpy as np
import typing
import xarray as xr
import glob
import os
import logging

def filter_files(files: typing.List[str], tile, year) -> typing.List[str]:
    """
    Filters a list of DEM file paths based on the specified tile and year.

    Parameters
    ----------
    files : list[str]
        A list of file paths to filter.
    tile : str
        The Sentinel-2 tile ID to filter files by.
    year : int
        The year to filter files by.

    Returns
    -------
    list[str]
        A list of file paths that match the specified filters.

    Notes
    -----
    The function assumes that DEM file names contain tile and year information 
    in specific formats.

    """

    filtered_files = []
    
    for file in files:
        name = os.path.basename(file)
        file_tile = name[:5]
        file_year = name[27:31]
        
        # Filter conditions
        matches_year = year is None or int(file_year) == int(year)
        matches_tile = tile is None or str(file_tile) == str(tile)
        
        # Filter based on year and tile of the composite
        if matches_year and matches_tile:
            filtered_files.append(file)
            
    return filtered_files
    
def apply_preprocess(
    ds: xr.Dataset,
    zip_file: str,
    preprocess: typing.Callable[[xr.Dataset], xr.Dataset] | None = None
) -> xr.Dataset:
    """
    Applies an optional preprocessing function to a DEM dataset and assigns
    metadata including the tile ID and file name.

    Parameters
    ----------
    ds : xr.Dataset
        The input DEM dataset to preprocess.
    zip_file : str
        The file name of the DEM zip file, used to extract metadata.
    preprocess : typing.Callable[[xr.Dataset], xr.Dataset] | None, optional
        A preprocessing function to apply to the dataset. If None, no
        preprocessing is performed.

    Returns
    -------
    xr.Dataset
        The DEM dataset with preprocessing applied (if specified) and additional
        metadata assigned as coordinates.

    Notes
    -----
    - The function ensures "tile" is a dimension in the dataset.
    - Assigns the tile ID and file name as coordinates for easier identification.
    - The function assumes that DEM file names contain tile and year information 
    in specific formats.
    """

    
    if preprocess:
        ds = preprocess(ds)
        
    zip_name = os.path.basename(zip_file)
    tile = zip_name[:5]
    
    ds = ds.squeeze()
    ds = ds.expand_dims("tile")
    ds = ds.assign_coords({"tile": [tile]})
    ds = ds.assign_coords({"file_name": ("tile", [zip_name])})

    return ds
    
def load_dems(
    path: typing.Union[str, pathlib.Path, list[str]],
    year : str,
    tile : str,
    preprocess: typing.Callable[[xr.Dataset], xr.Dataset] | None = None
):
    """Loads DEMs data from a given path or list of paths, optionally
    applying a preprocessing function and filtering the paths based on a requested tile.
    Year is actually not used in filtering logic.

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
        The loaded xarray dataset containing the DEMs requested.

    Raises
    ------
    ValueError
        If 'preprocess' is provided when reading a single file or if the
        'preprocess' argument is not applicable. Also if the filter leaves with no data to load.
        
    """  
    logger = logging.getLogger("load-dems")
    logger.info("Loading DEMs ...")
    
    if isinstance(path, str) and ("*" in path or "?" in path):
        path = glob.glob(str(path))  # Expand wildcard

    path = filter_files(path, tile, year)
    
    if not path:
        raise ValueError(f"No DEMs found for tile {tile} and year {year}")
        
    if isinstance(path, list):

        ds = xr.open_mfdataset(
            path,
            chunks="auto",
            engine="rasterio",
            preprocess=lambda ds: apply_preprocess(ds, os.path.basename(ds.encoding["source"])),
            parallel=True,
            combine="by_coords")
    
    else:
        
        # Single file case
        ds = xr.open_dataset(
            path,
            chunks="auto",
            engine="rasterio"
        )
        ds = apply_preprocess(ds, path)

    logger.info(f"Loaded {len(path)} files for tile {tile} and year {year}.")

    return ds

def calculate_slope_from_dems(dems: xr.Dataset) -> xr.Dataset:
    """Calculates the slope (steepness) in degrees from a Digital Elevation Model (DEM) dataset.

    Slope Calculation
    -----------------
    Slope represents how steep the terrain is. It is calculated by measuring how much the elevation
    changes from one point to its neighboring points in both horizontal (x) and vertical (y) directions.
    The result shows the steepness in degrees, with higher values indicating steeper slopes.

    Parameters
    ----------
    dems : xr.Dataset
        An xarray Dataset containing the DEM data, with spatial dimensions "y" and "x".

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the slope data in degrees, with the same spatial dimensions as the input.
    """
    logger = logging.getLogger("slope-calculation")
    logger.info("Calculating Slope ...")
    
    def compute_slope(dem_array: np.ndarray) -> np.ndarray:
        """Calculates slope by finding elevation changes between neighboring points."""
        dy, dx = np.gradient(dem_array.astype(np.float32))
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype("float32")
        return slope
    
    dems = dems.chunk({"y": -1, "x": -1})
    slope = xr.apply_ufunc(
        compute_slope,
        dems,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32]
    )
    
    logger.info("Slope Calculated.")
    return slope

def calculate_aspect_from_dems(dems: xr.Dataset) -> xr.Dataset:
    """Calculates the aspect (direction of slope) in degrees from a Digital Elevation Model (DEM) dataset.

    Aspect Calculation
    ------------------
    Aspect represents the compass direction that a slope faces. It is calculated by finding the direction
    in which the terrain has the steepest downward slope. The result is given in degrees, with values
    ranging from 0 to 360, where 0° represents North, 90° East, 180° South, and 270° West.

    Parameters
    ----------
    dems : xr.Dataset
        An xarray Dataset containing the DEM data, with spatial dimensions "y" and "x".

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the aspect data in degrees, with the same spatial dimensions as the input.
        Aspect values range from 0 to 360 degrees.
    """
    logger = logging.getLogger("aspect-calculation")
    logger.info("Calculating Aspects ...")
    
    def compute_aspect(dem_array: np.ndarray) -> np.ndarray:
        """Calculates aspect by finding the direction of the steepest slope."""
        dy, dx = np.gradient(dem_array.astype("float32"))
        aspect = np.degrees(np.arctan2(dy, -dx)).astype(np.float32)
        aspect = np.mod(450 - aspect, 360)  # Adjust aspect to range [0, 360]
        return aspect
    
    dems = dems.chunk({"y": -1, "x": -1})
    aspect = xr.apply_ufunc(
        compute_aspect,
        dems,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32]
    )
    
    logger.info("Aspects Calculated.")
    return aspect
