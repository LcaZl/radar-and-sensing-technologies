import xarray as xr
from multiprocessing import Pool, cpu_count
from ._utils import create_geodataframe
import numpy as np
import pandas as pd
import logging
import geopandas as gpd

def load_lc_points(
    parameters: dict,
) -> np.ndarray:
    """
    Loads land cover (LC) map.

    Parameters
    ----------
    parameters : dict
        Configuration settings containing the path to the LC map:
        - "lc_map_path": Path to the LC map file.

    Returns
    -------
    np.ndarray
        LC map as a NumPy array.
    """

    
    # Loaded LC map
    lc_map = xr.open_dataarray(parameters["lc_map_path"]).squeeze().values
    logger = logging.getLogger("load-lc-points")
    
    # Use vocabulary to convert all values in LC map loaded to id
    #convert = np.vectorize(lccode2id.get)
    #lc_map_conv = convert(lc_map)
    logger.info(f"Labels in original LC map: {np.unique(lc_map)}")
    #logger.info(f"Labels in converted LC map: {np.unique(lc_map_conv)}")
    
    return lc_map #lc_map_conv

def create_geodf_for_lc_map(
    classes_masks: dict,
    parameters: dict,
    original_points: gpd.GeoDataFrame,
    features: list,
    lccode2label: pd.DataFrame,
    composites_crs: str
) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame for land cover points, excluding test/validation points and converting labels.

    Parameters
    ----------
    classes_masks : dict
        Dictionary containing masks for each class.
    parameters : dict
        Configuration settings including:
        - "samples_per_class": Number of samples per class.
    original_points : gpd.GeoDataFrame
        GeoDataFrame of points to be excluded from sampling.
    features : list
        Features used for generating the GeoDataFrame.
    lccode2label : dict
        Dict containing label descriptions for mapping class IDs.
    composites_crs : str
        CRS for georeferencing the output GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Sampled LC points as a GeoDataFrame with class labels and CRS.
    """


    logger = logging.getLogger("create-geodf-for-lc")
    
    # Exclude point in test/validation dataset from LC points
    exclude_points = set(zip(original_points['x'], original_points['y']))
    
    logger.info(f"Classes: {len(classes_masks)} - Points: {len(classes_masks[list(classes_masks.keys())[0]])}")
    logger.info(f"Test/Val points count: {len(original_points)}")
    logger.info(f"Point of LC map excluded: {len(exclude_points)}")

    points = sample_points(classes_masks, exclude_points, samples_per_class=parameters["samples_per_class"])
    points["split"] = 'train'
    points['class'] = points['class_id'].map(lccode2label)

    points = create_geodataframe(points, features, composites_crs)
    
    logger.info("Points CRS:")
    print("Train Points CRS:", points.crs)
    print("Test Val Points CRS:", original_points.crs)
    
    return points

def sample_points(
    classes_masks: dict, 
    exclude_points: set, 
    samples_per_class: int = 1000
) -> pd.DataFrame:
    """
    Samples points from a 2D land cover map for each class, excluding specified points.

    Parameters
    ----------
    lc_map : np.ndarray
        2D array where each pixel value represents a class.
    exclude_points : set
        Set of (x, y) coordinates to exclude from sampling.
    samples_per_class : int, optional
        Number of points to sample per class (default is 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame with sampled points and class IDs (columns: x, y, class_id).
    """

    logger = logging.getLogger("sample-points")
    unique_classes = np.unique(list(classes_masks.keys()))
    
    logger.info("Starting parallel sampling of point...")
    logger.info(f"Excluding {len(exclude_points)} points from sampling.")
    logger.info(f"Found {len(unique_classes)} unique classes in the map.")
    
    # Prepare arguments for parallel processing
    args = [(class_id, classes_masks[class_id], exclude_points, samples_per_class) for class_id in unique_classes]
    
    # Use multiprocessing to sample points in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(sample_class_points, args)
    
    # Flatten the results and convert to DataFrame
    sampled_pixels = [pixel for result in results for pixel in result]
    logger.info(f"Sampling complete. Total sampled points: {len(sampled_pixels)}")
    
    return pd.DataFrame(sampled_pixels)

def sample_class_points(
    class_id: int, 
    class_mask: np.ndarray, 
    exclude_points: set, 
    samples_per_class: int
) -> list:
    """
    Samples points for a specific class using systematic sampling.

    Parameters
    ----------
    class_id : int
        Class ID to sample.
    lc_map : np.ndarray
        2D land cover map array.
    exclude_points : set
        Set of (x, y) coordinates to exclude.
    samples_per_class : int
        Number of points to sample for the class.

    Returns
    -------
    list
        List of dictionaries with sampled points (x, y, class_id).
    """

    logger = logging.getLogger("samples-for-class")
    logger.info(f"Processing class {class_id}...")
    
    # Get all coordinates for the current class
    class_coords = np.argwhere(class_mask)
    
    # Remove excluded points
    class_coords = [tuple(coord) for coord in class_coords if tuple(coord) not in exclude_points]
    
    # Ensure there are enough points for sampling
    total_available = len(class_coords)
    
    if total_available < samples_per_class:
        
        logger.info(f"Warning: Class {class_id} has only {total_available} points, less than the requested {samples_per_class}.")
        samples_per_class = total_available
    
    # Determine sampling interval
    interval = max(1, total_available // samples_per_class)
    
    # Select points using systematic sampling
    sampled_coords = class_coords[::interval][:samples_per_class]
    
    # Log sampling results
    logger.info(f"Class {class_id}: Sampled {len(sampled_coords)} points (requested {samples_per_class}).")
    
    # Format as a list of dictionaries
    return [{"x": coord[0], "y": coord[1], "class_id": class_id} for coord in sampled_coords]



