import datetime
import pathlib
import typing
import os
import logging
import glob
import re
import xarray as xr
import numpy as np

from ._utils import get_season_id

def load_s2(
    path: list[str | pathlib.Path] | str | pathlib.Path,
    group: str,
    preprocess: typing.Callable[[xr.Dataset], xr.Dataset] | None = None,
    year : int = None,
    tile : str = None,
    sensor : str = None
):

    """
    Loads Sentinel-2 data from a given path or list of paths, optionally
    applying a preprocessing function. Supports loading single or multiple
    files with xarray.

    Parameters
    ----------
    path : list[str  |  pathlib.Path] | str | pathlib.Path
        The file path(s) to the Sentinel-2 data. Can be a single path, a list
        of paths, or a path pattern with wildcards.
    group : str
        The specific group within the dataset to load.
    preprocess : typing.Callable[[xr.Dataset], xr.Dataset] | None
        An optional function to preprocess the data after loading. This is only
        applicable when loading multiple files.
    year : str
        Filter products on requested year, if available.
    tile : str
        Filter products on requested tile, if available.
    sensor : str
        Filter products on requested sensor, if available.

    Returns
    -------
    _type_
        The loaded xarray dataset containing the Sentinel-2 data.

    Raises
    ------
    ValueError
        If `preprocess` is provided when reading a single file or if the
        `preprocess` argument is not applicable.
    """
    logger = logging.getLogger("load-s2")
    # Expand the wildcard paths
    if isinstance(path, str) and ("*" in path or "?" in path):
        path = glob.glob(str(path))  # Expand the wildcard to get a list of actual file paths

    # Function to filter files by year and tile
    def filter_files(files: typing.List[str]) -> typing.List[str]:
        filtered_files = []

        for file in files:

            zip_name = os.path.basename(file)
            file_tile = zip_name[39:44]
            match = re.search(r'\d{8}', zip_name)
            first_date = match.group() if match else None
            file_year = first_date[:4]
            file_month = first_date[4:6]
            file_sensor = zip_name[:3]


            # Filter based on year and tile of the composite
            if int(file_year) == int(year) and file_tile == tile and sensor in file_sensor:
                filtered_files.append(file)

        return filtered_files
    
    path = filter_files(path)

    if not path:
        raise ValueError(f"No files founded for tile {tile} and year {year}")
        
    def apply_preprocess(ds: xr.Dataset, zip_file: str) -> xr.Dataset:
        """Applies the preprocessing function."""
        if preprocess:
            ds = preprocess(ds)
        
        # To have a correct and related to dataset output path for the output.
        zip_name = os.path.basename(zip_file)
        tile_id = zip_name[39:44]

        season_idx = f"{tile_id}{get_season_id(ds.time.values[0])}"
        
        ds = ds.assign_coords({"file_name": ("time", [zip_name[4:]])})
        ds = ds.assign_coords({"season_id": ("time", [season_idx])})
        ds = ds.assign_attrs({"tile_id":tile_id})
        
        return ds

    # Handle multiple files (expanded paths from wildcard) using xarray's `open_mfdataset`
    if isinstance(path, list):
        
    
        # Ensure each file gets its own zip name during preprocessing
        # Load datasets with Dask and apply preprocessing
        ds = xr.open_mfdataset(
            path,
            chunks="auto",
            engine="rasterio",
            group=group,
            preprocess=lambda ds: apply_preprocess(ds, os.path.basename(ds.encoding["source"])),
            parallel=True,  # Enable parallel loading
            combine='by_coords'  # Ensures that datasets are combined correctly along coordinates like 'time'
        )
    
    else:
        
        # Handle the case for single files (preprocess manually after loading)
        ds = xr.open_dataset(
            path,
            chunks="auto",
            engine="rasterio",
            group=group
        )
        
        logger.info(f"Loading dataset from: {path}")
        ds = apply_preprocess(ds, path)

    logger.info(f"Dataset (resolution: {group}) loaded - Dimensions: {list(ds.dims)} - Coordinates: {list(ds.variables)}")
    return ds

def preprocess(
    xds: xr.Dataset,
) -> xr.Dataset:
    """Preprocess a Sentinel-2 dataset or data array by removing the offset
    values from the diginal numbers (DNs) if present and expands the dataset
    by adding a time dimension based on the acquisition time.

    Parameters
    ----------
    xda : xr.Dataset | xr.DataArray
        The input xarray dataset or data array representing the Sentinel-2 data.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The preprocessed xarray dataset or data array with the offset removed
        from the DNs (if applicable) and a time dimension added.
    """
    acq_time = datetime.datetime.strptime(
        xds.PRODUCT_START_TIME.split(".")[0], "%Y-%m-%dT%H:%M:%S"
    )

    # Sentinel-2 PB >= 4.00 has offset values to DNs. Removing it when present
    var_name = list(xds.keys())[0]
    offset = xds[var_name].attrs.get("BOA_ADD_OFFSET", 0)
    bands = [b.split(",")[0] for b in xds[var_name].attrs["long_name"] if b[0] == "B"]
    xds[dict(band=slice(len(bands)))] = (
        xds[dict(band=slice(len(bands)))].clip(-offset) + offset
    )

    xds[var_name] = xds[var_name].astype(np.uint16)

    xds = xds.rename({var_name: "data"})

    return xds.expand_dims(time=[acq_time])

######################################
#     Retrieve masks from data       #
######################################

def get_scl_mask(scl_xda: xr.Dataset, scl_to_mask: list[int]) -> xr.DataArray:
    """Generate a mask for specified Sentinel-2 Scene Classification Layer (SCL)
    values in the given variable in a xarray dataset.

    Parameters
    ----------
    scl_xda : xr.Dataset
        The input xarray dataset containing the Sentinel-2 SCL data in a variable.
    scl_to_mask : list[int]
        A list of SCL values to be masked. The mask will be True for these values
        and False otherwise.

    Returns
    -------
    xr.DataArray
        An xarray data array containing the generated mask, where the specified
        SCL values are marked as True.
    """
    scl_mask = scl_xda["data"] == scl_to_mask[0]
    for v in scl_to_mask[1:]:
        scl_mask = scl_mask | (scl_xda.data == v)
        
    
    return scl_mask

######################################
#   Sentinel 2 bands management      #
######################################

def drop_aux_bands(**dss: xr.Dataset) -> tuple[dict[str, xr.Dataset], list[str]]:
    """Extracts and returns a list of band names from the 10m, 20m, and 60m
    resolution datasets after filtering out auxiliary band names.

    Parameters
    ----------
    dss : dict[str, xr.Dataset]
        A dict of xarray datasets containing band data with non-spectral bands.

    Returns
    -------
    tuple[dict[str, xr.Dataset], list[str]]
        A tuple of a dict of xarray datasets containing band data only and a
        list of band names final datasets.
    """
    cleaned_dss: dict[str, xr.Dataset] = dict()
    band_names: list[str] = []
    for res, ds in dss.items():

        ds.variables[list(ds.keys())[0]].attrs["long_name"] = [
            b
            for b in ds.variables[list(ds.keys())[0]].attrs["long_name"]
            if b[0] == "B"
        ]
        cleaned_dss[res] = ds.sel(band=[b for b in ds.band.to_numpy() if b[0] == "B"])
        band_names += ds.variables[list(ds.keys())[0]].attrs["long_name"]

    return cleaned_dss, band_names
    
def set_bands(xda: xr.Dataset, only_bands: bool = False) -> xr.Dataset:
    """Sets the band names for an xarray dataset and optionally filters only the
    bands that start with 'B' (no auxiliary data). Updates the dataset with the
    new band names.

    Parameters
    ----------
    xda : xr.Dataset
        The input xarray dataset representing the Sentinel-2 data.
    only_bands : bool, optional
        If True, only the bands that start with 'B' are retained in the dataset,
        by default False.

    Returns
    -------
    xr.Dataset
        The xarray dataset with updated band names and, if 'only_bands' is True,
        filtered to include only the relevant bands.
    """
    bands = [
        b.split(",")[0] for b in xda.variables[list(xda.keys())[0]].attrs["long_name"]
    ]
    if only_bands:
        bands = [b for b in bands if b[0] == "B"]
        xda = xda.isel(band=slice(len(bands)))
    xda["band"] = bands
    return xda