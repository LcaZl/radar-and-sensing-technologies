import logging
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import CRS
from ._output_presentation import print_dataframe

def load_points(
    parameters: dict, 
    features: xr.Dataset, 
    composites_crs: str
) -> tuple:
    """
    Loads and processes points for training, testing and validation, filtering by tile and matching coordinates.

    Parameters
    ----------
    parameters : dict
        Configuration settings including paths for points dataset, labels, and KML grid.
    features : xr.Dataset
        Xarray dataset containing the feature composites.
    composites_crs : str
        CRS for georeferencing the points to match the composites dataset.

    Returns
    -------
    tuple
        - gpd.GeoDataFrame: Filtered and matched points for the current tile.
        - pd.DataFrame: DataFrame containing label descriptions.
        - shapely.geometry: Geometry of the tile used for filtering.
    """

    logger = logging.getLogger("load-points")

    # Load points for testing and validation
    points_df = gpd.read_file(f"zip://{parameters['points_dataset_path']}")
    labels_df = pd.read_csv(parameters["labels_path"])

    # Adding a column with the label string
    points_df['class'] = points_df['Level_1'].map(labels_df['description'])
    points_df.rename(columns = {"Level_1":"class_id"}, inplace=True)

    print(points_df.geometry)
    # Extract tile coordinates using KML grid file
    tile_df = get_tile_geometry_from_kml(parameters["kml_path"], parameters["tile_id"])
    tile_geometry = tile_df.geometry

    # Filter loaded points for the current tile
    print(f"Points: {len(points_df)}")
    points_df_filtered = points_df[points_df["geometry"].within(tile_geometry.iloc[0].geoms[0])]
    print(f"Points: {len(points_df_filtered)}")
    
    points_df_filtered = points_df_filtered.to_crs(epsg=composites_crs) 

    # Match them to coordinates usable with xarray isel(), for fast indexing
    # And store them with each sample
    print(f"Shapefile poiints CRS: {points_df_filtered.crs}")
    points_matched = match_points_to_dataset(points_df_filtered, features)
    points_matched["split"] = 'test'

    # Inspect CRS of xarray composite/features dataset and the training points geo dataframe
    logger.info(f"Dataset CRS: EPSG:{CRS.from_wkt(features.spatial_ref.attrs['crs_wkt']).to_epsg()}")
    logger.info(f"Training Points CRS: {points_matched.crs}")
    
    if parameters["verbose"] == True:
        print_dataframe(labels_df, title="\nLabels")
        print_dataframe(points_matched, title="\nTraining points")
    
    return points_matched, labels_df, tile_geometry

def get_tile_geometry_from_kml(
    kml_path: str, 
    tile_id: str
) -> gpd.GeoDataFrame:
    """
    Extracts the geometry of a specific Sentinel-2 tile from a KML file.

    Parameters
    ----------
    kml_path : str
        Path to the Sentinel-2 KML file.
    tile_id : str
        Tile ID to filter (e.g., "21KUQ").

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the geometry of the specified tile.
    """

    
    logger = logging.getLogger("get-tile-geometry")
    
    # Load the KML file
    kml_data = gpd.read_file(kml_path, driver="KML")
    print(f"KML file columns: {kml_data.keys()}")

    # Filter for the specific tile
    tile_geometry = kml_data[kml_data['Name'] == tile_id]

    if tile_geometry.empty:
        raise ValueError(f"Tile {tile_id} not found in the KML file.")

    return tile_geometry

from pyproj import Transformer
from affine import Affine

def match_points_to_dataset(
    points_projected: gpd.GeoDataFrame, 
    dataset: xr.Dataset
) -> gpd.GeoDataFrame:
    """
    Matches points to pixel indices in a dataset using affine transform from raster metadata.
    This method ensures correct geospatial alignment by reconstructing the pixel transform.

    Parameters
    ----------
    points_projected : gpd.GeoDataFrame
        GeoDataFrame with geometries already reprojected to match the dataset CRS.
    dataset : xr.Dataset
        Xarray dataset with 'x' and 'y' coordinates and spatial_ref.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with additional columns ('x', 'y') as integer pixel indices.
    """

    # Get spatial info from xarray dataset
    x_coords = dataset.coords['x'].values
    y_coords = dataset.coords['y'].values

    # Infer resolution and origin
    res_x = x_coords[1] - x_coords[0]
    res_y = y_coords[1] - y_coords[0]

    origin_x = x_coords[0]
    origin_y = y_coords[0]

    # Define the affine transform
    # Note: for y we use negative resolution since y usually decreases from top to bottom
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(res_x, -abs(res_y))

    # Get coordinates
    point_x = points_projected.geometry.x.values
    point_y = points_projected.geometry.y.values

    # Apply inverse affine transform to get pixel indices
    pixel_coords = [~transform * (x, y) for x, y in zip(point_x, point_y)]
    pixel_x_indices, pixel_y_indices = zip(*[(int(round(px)), int(round(py))) for px, py in pixel_coords])

    # Add to GeoDataFrame
    points_projected['x'] = pixel_x_indices
    points_projected['y'] = pixel_y_indices

    return points_projected