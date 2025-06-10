import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from shapely.geometry import Point

def filter_dict_by_prefix(input_dict: dict, prefix: str) -> dict:
    """
    Filters dictionary keys by a given prefix and removes the prefix from the keys.

    Parameters
    ----------
    input_dict : dict
        Dictionary to filter.
    prefix : str
        Prefix to filter by.

    Returns
    -------
    dict
        Filtered dictionary with prefix removed from keys.
    """
    return {k.replace(prefix, '') : v for k, v in input_dict.items() if k.startswith(prefix)}

def df_numerical_columns_stats(df: pd.DataFrame, title: str = "") -> None:
    """
    Prints basic statistics for numerical columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numerical columns to analyze.
    title : str, optional
        Title to display before the statistics (default is "").
    """
    if title:
        print(title)
    for i, key in enumerate(df.select_dtypes(include=['int', 'float', 'double']).columns):
        print(f"{i} - {key} - Min: {round(df[key].min(), 3)} - Max: {round(df[key].max(), 3)} - Mean: {round(df[key].mean(), 3)} - Std: {round(df[key].std(), 3)} - Var: {round(df[key].var(), 3)}")

def create_geodataframe(
    df: pd.DataFrame, 
    dataset: xr.Dataset, 
    crs: str, 
    x_col: str = 'x', 
    y_col: str = 'y'
) -> gpd.GeoDataFrame:
    """
    Converts a DataFrame to a GeoDataFrame using pixel coordinates and dataset georeferencing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with pixel coordinates.
    dataset : xr.Dataset
        Xarray dataset containing spatial reference and GeoTransform.
    crs : str
        Coordinate Reference System for georeferencing.
    x_col : str, optional
        Column name for pixel x-coordinates (default is 'x').
    y_col : str, optional
        Column name for pixel y-coordinates (default is 'y').

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with 'geometry' column containing Point objects.
    """

    # Apply pixel_to_latlon to compute lat, lon for each row
    def compute_geometry(row):
        lon, lat = pixel_to_latlon(dataset, row[x_col], row[y_col])
        return Point(lon, lat)

    # Create the geometry column
    df['geometry'] = df.apply(compute_geometry, axis=1)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf = gdf.set_crs(epsg=crs) 
    
    return gdf

def pixel_to_latlon(dataset: xr.Dataset, pixel_x: int, pixel_y: int) -> tuple:
    """
    Converts pixel coordinates to geographic coordinates using GeoTransform.

    Parameters
    ----------
    dataset : xr.Dataset
        Xarray dataset with spatial reference containing GeoTransform attributes.
    pixel_x : int
        Pixel x-coordinate.
    pixel_y : int
        Pixel y-coordinate.

    Returns
    -------
    tuple
        Longitude and latitude as (lon, lat).
    """

    # Parse GeoTransform from spatial_ref attributes
    geo_transform_str = dataset.spatial_ref.attrs["GeoTransform"]
    geo_transform = [float(val) for val in geo_transform_str.split()]
    
    # Calculate geographic coordinates
    lon = geo_transform[0] + (pixel_x + 0.5) * geo_transform[1] + (pixel_y + 0.5) * geo_transform[2]
    lat = geo_transform[3] + (pixel_x + 0.5) * geo_transform[4] + (pixel_y + 0.5) * geo_transform[5]
    return lon, lat


def get_season_id(time: np.datetime64 | pd.Timestamp) -> str:
    """
    Generates a unique identifier for the season in the format 'YYYY_SSN'.

    Parameters
    ----------
    time : np.datetime64 or pd.Timestamp
        Input time as a datetime object.

    Returns
    -------
    str
        Season identifier in the format 'YYYY_SSN'.
    """

    
    month = time.astype('datetime64[M]').astype(int) % 12 + 1
    year = time.astype('datetime64[Y]').astype(int) + 1970
    return f"{str(year)}_{get_season(month)}"
        
        
def get_season(month: int) -> str:
    """
    Returns the season corresponding to the input month.

    Parameters
    ----------
    month : int
        Month as an integer (1 = January, ..., 12 = December).

    Returns
    -------
    str
        Season identifier:
        - 'DJF' for Winter (Dec, Jan, Feb)
        - 'MAM' for Spring (Mar, Apr, May)
        - 'JJA' for Summer (Jun, Jul, Aug)
        - 'SON' for Fall (Sep, Oct, Nov)
    """

    
    if month in [1, 2, 3]:
        return 1  # Winter (December, January, February)
    elif month in [4, 5, 6]:
        return 2  # Spring (March, April, May)
    elif month in [7, 8, 9]:
        return 3  # Summer (June, July, August)
    else:
        return 4  # Fall (September, October, November)
