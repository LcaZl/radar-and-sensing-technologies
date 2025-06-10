import numpy as np
import xarray as xr
import pandas as pd
import logging
from shapely.geometry import GeometryCollection
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import cv2
import geopandas as gpd

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import pickle

def convert_sav_to_csv(
    sav_path: str, 
    csv_path: str, 
    parameters: dict, 
    n_months: int = 12, 
    n_bands: int = 12, 
    glcm_names: list[str] = None
) -> pd.DataFrame:
    """
    Converts a serialized .sav dataset file into a structured CSV format with labeled features.

    Parameters
    ----------
    sav_path : str
        Path to the input .sav file containing the feature dataset and labels.
    csv_path : str
        Path where the output CSV file will be saved.
    parameters : dict
        Dictionary of runtime parameters. Must include 'labels_path', pointing to a CSV file
        with LC class metadata (columns: 'internal_code', 'LC_code', 'description').
    n_months : int, optional
        Number of months used to generate the band features, default is 12.
    n_bands : int, optional
        Number of spectral bands per month, default is 12.
    glcm_names : list of str, optional
        List of GLCM texture feature names. The default list of six standard features
        is: ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'].

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the full structured dataset, with named columns and labels.

    Notes
    -----
    - The .sav file must contain two objects in sequence: a 2D array of feature values and a label array.
    - The resulting DataFrame includes:
        - Flattened monthly band features (e.g., band_1_month_1, ..., band_12_month_12),
        - GLCM texture features,
        - Ground truth indices (mapped to LC codes),
        - Human-readable LC labels,
        - A default split assignment of 'test' for compatibility.
    """
    if glcm_names is None:
        glcm_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']


    
    with open(sav_path, 'rb') as file:
        dataset = pickle.load(file)
        labels = pickle.load(file)

    # Build static header
    feature_names = [f"band_{b}_month_{m}" for m in range(1, n_months + 1) for b in range(1, n_bands + 1)]
    feature_names += [f"GLCM{name}" for name in glcm_names]
    feature_names.append("ground_truth_index")

    df = pd.DataFrame(dataset)
    df["ground_truth_index"] = labels.astype(int)
    df.columns = feature_names
    
    labels_df = pd.read_csv(parameters["labels_path"])
    id2lccode = {row['internal_code'] : row['LC_code'] for id, row in labels_df.iterrows()}
    lccode2label = {row['LC_code'] : row['description'] for id, row in labels_df.iterrows()}
    
    #print(df["ground_truth_index"].value_counts())
    
    df["ground_truth_index"].value_counts()
    df["ground_truth_index"] = df["ground_truth_index"].map(id2lccode)
    df["ground_truth_label"] = df["ground_truth_index"].map(lccode2label)
    df["split"] = 'test'

    df.to_csv(csv_path, index=False)

    return df

def extract_features_for_points(
    dataset: xr.Dataset, 
    points: gpd.GeoDataFrame, 
    path: str
) -> None:
    """
    Extracts features for each point from the dataset, including spectral bands and additional features.
    Saves the dataset to a CSV file.

    Parameters
    ----------
    dataset : xr.Dataset
        Xarray dataset containing band data and additional variables.
    points : gpd.GeoDataFrame
        GeoDataFrame containing point coordinates, class IDs, and labels.
    path : str
        Path to save the dataset as a CSV file.

    Returns
    -------
    None
    """
    pixel_x = points['x'].astype(int).values
    pixel_y = points['y'].astype(int).values
    ground_truth_index = points['class_id'].values
    ground_truth_label = points['class'].values
    split = points['split'].values
    
    # Create DataArrays for x and y indices
    index_x = xr.DataArray(pixel_x, dims="points")
    index_y = xr.DataArray(pixel_y, dims="points")

    # Flatten the band_data (12 bands x 12 months = 144 features per point)
    band_data_flattened = (
        dataset['band_data']
        .isel(x=index_x, y=index_y)
        .stack(features=("band", "time"))
        .values
    )

    # Extract single values from other variables for each point (7 GLCM features + NDI + NDVI = 9)
    additional_features = {
        var: dataset[var].isel(x=index_x, y=index_y).values
        for var in dataset.data_vars if var != 'band_data'
    }

    # Combine all features into a single matrix
    training_matrix = np.column_stack([
        pixel_x,
        pixel_y,
        ground_truth_index,
        ground_truth_label,
        split,
        band_data_flattened,
        *[additional_features[var] for var in additional_features]
    ])

    # Define column names
    columns = (
        ['x', 'y', 'ground_truth_index', 'ground_truth_label', 'split'] +
        [f"band_{band}_month_{month}" for band in range(1, 13) for month in range(1, 13)] +
        list(additional_features.keys())
    )

    training_df = pd.DataFrame(training_matrix, columns=columns)
    training_df.to_csv(path, index=False)
    return training_df



def enhance_dataset(
    samples_to_enhance: pd.DataFrame, 
    samples_for_enhance: pd.DataFrame, 
    target_class_col: str, 
    samples_per_class: int, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Enhances the target dataset by adding samples from a source land cover map to balance class distribution.

    Parameters
    ----------
    samples_to_enhance : pd.DataFrame
        Target dataset to enhance.
    samples_for_enhance : pd.DataFrame
        Source dataset for additional samples.
    target_class_col : str
        Column name representing class labels.
    samples_per_class : int
        Target number of samples per class.
    verbose : bool, optional
        If True, displays debug messages.

    Returns
    -------
    pd.DataFrame
        Enhanced dataset with balanced class distribution.
    """

    # Count current class distribution
    logger = logging.getLogger("enhance-dataset")
    class_counts = samples_to_enhance[target_class_col].value_counts()

    if verbose:
        logger.info("Enhance Dataset Info")
        logger.info(f"Initial class distribution:\n{class_counts}")
        logger.info(f"Target samples per class: {samples_per_class}\n")

    # DataFrame to hold additional samples
    additional_samples = []

    for class_id, count in class_counts.items():
        if count < samples_per_class:
            # Number of samples to add
            samples_needed = samples_per_class - count

            # Get samples for this class
            class_samples = samples_for_enhance[samples_for_enhance[target_class_col] == class_id]

            if not class_samples.empty:
                # Randomly sample from the available class samples
                sampled = class_samples.sample(n=samples_needed, replace=len(class_samples) < samples_needed)
                additional_samples.append(sampled)

                if verbose:
                    logger.info(f"Class {class_id}: {count} -> {samples_per_class} (+{samples_needed} samples)")
            else:
                logger.info(f"Class {class_id}: No additional samples available in samples_for_enhance. Skipping.")

    # Concatenate the additional samples to the original dataset
    if additional_samples:
        enhanced_dataset = pd.concat([samples_to_enhance, pd.concat(additional_samples)], ignore_index=True)
        if verbose:
            logger.info(f"Final class distribution:\n{enhanced_dataset[target_class_col].value_counts()}")
    else:
        enhanced_dataset = samples_to_enhance  # No additional samples needed
        logger.info("No enhancements were made. The dataset remains unchanged.")
    logger.info("Dataset enhancement completed successfully.")

    return enhanced_dataset

def apply_erosion_and_report(
    lc_map_xr: np.ndarray, 
    lccode2label: dict, 
    config: dict, 
    output_path: str = None
) -> dict:
    """
    Applies morphological erosion to each class mask and generates a visual report.

    Parameters
    ----------
    lc_map_xr : np.ndarray
        Land cover classification map as a NumPy array.
    id2label : dict
        Dictionary mapping class IDs to class names.
    config : dict
        Configuration parameters including:
        - 'kernel': Tuple for erosion kernel size.
        - 'iterations': Number of iterations for erosion.
    output_path : str, optional
        Path to save the erosion report as an image (default is None).

    Returns
    -------
    dict
        Dictionary containing eroded masks for each class.
    """


    # Get configuration parameters or set default values
    logger = logging.getLogger("erode-class-mask")
    kernel = config['kernel']
    iterations = config['iterations']

    # Dictionary to store eroded masks
    eroded_masks = {}
    plot_data = []

    # Iterate over each class
    for cls, class_name in lccode2label.items():

        logger.info(f"Processing Class {cls} - {class_name}")

        # Create binary mask for the class
        mask = (lc_map_xr == cls).astype(np.uint8)
        logger.info(f"Mask distribution: {np.unique(mask, return_counts=True)}")

        # Skip empty masks
        if mask.mean() == 0:
            logger.info(f"Class {cls} - {class_name} is empty, skipping.")
            continue
        
        # Apply erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
        logger.info("Eroded Mask distribution:")
        logger.info(np.unique(eroded_mask, return_counts=True))
        
        # Store the eroded mask
        eroded_masks[cls] = eroded_mask
        
        # Store for plotting
        plot_data.append((mask, eroded_mask, class_name))
    
    if output_path is not None:
        # Generate and save the report
        num_classes = len(plot_data)
        fig, axs = plt.subplots(num_classes, 2, figsize=(12, num_classes * 6))
        
        for i, (orig_mask, eroded_mask, class_name) in enumerate(plot_data):
            # Original Mask
            axs[i, 0].imshow(orig_mask, cmap='gray')
            axs[i, 0].set_title(f"Original Mask - {class_name}", fontsize = 18)
            axs[i, 0].axis('off')
            
            # Eroded Mask
            axs[i, 1].imshow(eroded_mask, cmap='gray')
            axs[i, 1].set_title(f"Eroded Mask - {class_name}", fontsize = 18)
            axs[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Report saved at: {output_path}")
        
    return eroded_masks