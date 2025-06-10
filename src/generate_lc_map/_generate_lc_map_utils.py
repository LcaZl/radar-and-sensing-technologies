import dask.array
import dask.delayed
import xarray as xr
import pandas as pd
from svm_pipeline import SVMS, load_SVMS
import dask.array as da
import numpy as np
import logging
import rioxarray
import matplotlib.pyplot as plt
import os
import rasterio

from joblib import Parallel, delayed
import numpy as np
import logging
import copy
import gc
from svm_pipeline import load_SVMS

def process_chunk(
    features_array: np.ndarray, 
    feature_names: list[str], 
    scalers: dict[str, object], 
    args: dict, 
    x_start: int,
    y_start: int,
    output_paths: dict[str, str],
    svms_to_use: list[str]
) -> None:
    """
    Processes a spatial chunk of input features, performs preprocessing, optional PCA projection,
    and computes predictions using the specified SVMS model for multiple classification strategies.
    Writes both predicted labels and class probabilities to the corresponding GeoTIFF files.

    Parameters
    ----------
    features_array : np.ndarray
        Array of shape (n_features, patch_x, patch_y) containing the feature values for the chunk.
    feature_names : list of str
        Names of the features corresponding to the first axis of features_array.
    scalers : dict of str to object
        Dictionary mapping feature names to fitted scalers for preprocessing.
    args : dict
        Additional arguments including:
            - 'svm': the SVMS model object
            - 'pca': optional PCA model
            - 'label_mapper': function to convert internal class indices to label codes
            - 'task_id': unique identifier for the current chunk task
    x_start : int
        Horizontal offset in the original full image.
    y_start : int
        Vertical offset in the original full image.
    output_paths : dict of str to str
        Dictionary mapping prediction keys to file paths for GeoTIFF output files.

    Returns
    -------
    None
        The function performs its tasks via side effects, writing predictions and probabilities
        directly into the corresponding GeoTIFF files.
    """

    
    try:
        logger = logging.getLogger("process-chunk")
        task_id = args['task_id']
        logger.info(f"Started task {task_id} - (X: {x_start}, Y: {y_start})")

        patch_x, patch_y = features_array.shape[1:]
        svm = copy.deepcopy(args["svm"])


        # Step 1: Serial Scaling
        preprocessed = np.zeros_like(features_array)
        for i, fname in enumerate(feature_names):
            preprocessed[i] = scalers[fname].transform(features_array[i].reshape(-1, 1)).reshape(patch_x, patch_y)


        # Step 2: PCA (optional)
        if svm.pca is not None:
            pca = args['pca']
            pca_indices = [feature_names.index(f) for f in pca.feature_names_in_]
            
            flat_pca_input = preprocessed[pca_indices].reshape(len(pca_indices), -1).T
            flat_pca_output = pca.transform(flat_pca_input)
            pca_components = flat_pca_output.T.reshape(pca.n_components_, patch_x, patch_y)

            # Ricostruzione dizionario
            preprocessed_dict = {fname: preprocessed[feature_names.index(fname)] for fname in feature_names if fname not in pca.feature_names_in_}
            preprocessed_dict.update({f"PC{i+1}": pca_components[i] for i in range(pca.n_components_)})

            # Ordina secondo svm.multiclass_svm.feature_names_in_
            final_order = svm.multiclass_svm.feature_names_in_
            preprocessed = np.stack([preprocessed_dict[fname] for fname in final_order], axis=0)


        # Step 3: Reformat for row-wise prediction
        # final shape: (patch_y, patch_x, n_features)
        reshaped = np.moveaxis(preprocessed, 0, -1)
        label_mapper = args['label_mapper']
        n_classes = len(svm.classes)

        def predict_and_write(svm_model, pred_key):
            
            def predict_row(y_idx):
                row_feats = reshaped[y_idx]  # shape: (patch_x, n_features)
                
                df_feats = pd.DataFrame(row_feats, columns=svm_model.multiclass_svm.feature_names_in_)
                svm_model.multiclass_svm

                labels = svm_model.predict(df_feats)
                probs = svm_model.predict_proba(df_feats)
                return labels, probs

            results = Parallel(n_jobs=4)(
                delayed(predict_row)(y) for y in range(patch_y)
            )

            # Rebuild matrices
            pred_labels = np.stack([r[0] for r in results], axis=0)  # (patch_y, patch_x)
            pred_probs = np.stack([r[1] for r in results], axis=0)   # (patch_y, patch_x, n_classes)

            pred_labels = label_mapper(pred_labels)

            # Write labels
            write_chunk_to_tif(output_paths[pred_key], pred_labels.astype(np.uint8), n_classes, x_start, y_start)

            # Write probabilities
            for class_idx in range(n_classes):
                prob_band = (pred_probs[:, :, class_idx] * 255).astype(np.uint8)
                write_chunk_to_tif(output_paths[pred_key], prob_band, class_idx, x_start, y_start)

            del pred_labels, pred_probs, results

        # Prediction 1: Multiclass SVM
        if 'svmMc' in svms_to_use:
            logger.info(f"Predicting with multiclass SVM")
            svm_mc = copy.deepcopy(svm)
            svm_mc.use_binary_svms = False
            svm_mc.use_multiclass_svm_calibrated = False
            predict_and_write(svm_mc, "svmMc")
        
        # Prediction 2: Calibrated multiclass
        if 'svmMcCal' in svms_to_use:
            logger.info(f"Predicting with multiclass SVM calibrated")
            svm_mc_cal = copy.deepcopy(svm)
            svm_mc_cal.use_binary_svms = False
            svm_mc_cal.use_multiclass_svm_calibrated = True
            predict_and_write(svm_mc_cal, "svmMcCal")

        # Prediction 3: Binary SVM
        if 'svmsBin' in svms_to_use:
            logger.info(f"Predicting with binary SVM")
            svm_bin = copy.deepcopy(svm)
            svm_bin.use_binary_svms = True
            svm_bin.use_binary_svms_with_softmax =  False
            svm_bin.use_multiclass_svm_calibrated = False
            predict_and_write(svm_bin, "svmsBin")
            
        # Prediction 4: Binary SVM with Softmax
        if 'svmsBinSoftmax' in svms_to_use:
            logger.info(f"Predicting with binary SVM with softmax")
            svm_bin = copy.deepcopy(svm)
            svm_bin.use_binary_svms = True
            svm_bin.use_binary_svms_with_softmax =  True
            svm_bin.use_multiclass_svm_calibrated = False
            predict_and_write(svm_bin, "svmsBinSoftmax")
            
        # Cleanup
        del features_array, preprocessed, reshaped, svm_mc #, svm_mc_cal, svm_bin
        gc.collect()
        logger.info(f"Ended task {task_id} - (X: {x_start}, Y: {y_start})")
    
    except Exception as e:
        logger.error(f"Error in task {args['task_id']} - (X: {x_start}, Y: {y_start})")
        logger.error(e)
        raise e
    
    
def write_chunk_to_tif(
    file_path: str,
    data: np.ndarray,
    band_index: int,
    x_start: int,
    y_start: int
) -> None:
    """
    Writes a 2D array (chunk) into a specific band and window of an existing GeoTIFF file.

    Parameters
    ----------
    file_path : str
        Path to the GeoTIFF file where the chunk will be written.
    data : np.ndarray
        2D array of shape (height, width) representing the data to write.
    band_index : int
        Index of the band to write into (0-based, automatically converted to 1-based for rasterio).
    x_start : int
        Horizontal starting coordinate of the chunk in the target image.
    y_start : int
        Vertical starting coordinate of the chunk in the target image.

    Returns
    -------
    None
    """

    with rasterio.open(file_path, 'r+') as dst:
        window = rasterio.windows.Window(x_start, y_start, data.shape[1], data.shape[0])
        dst.write(data, band_index + 1, window=window)  # +1 perché rasterio usa 1-based indexing

def create_empty_tif(
    path: str,
    width: int,
    height: int,
    num_bands: int,
    dtype: str,
    transform,
    crs,
    band_names: list = None
) -> None:
    """
    Creates a new empty GeoTIFF image with the specified spatial dimensions and metadata.

    Parameters
    ----------
    path : str
        Output file path for the GeoTIFF.
    width : int
        Width (number of columns) of the raster.
    height : int
        Height (number of rows) of the raster.
    num_bands : int
        Number of bands to include in the file.
    dtype : str
        Data type of pixel values (e.g., 'uint8', 'float32').
    transform : Affine
        Affine transformation mapping pixel to spatial coordinates.
    crs : rasterio.crs.CRS
        Coordinate reference system for the output raster.
    band_names : list, optional
        List of string descriptions for each band.

    Returns
    -------
    None

    Notes
    -----
    - If `band_names` is provided, it will be saved as band metadata.
    """   
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        if band_names is not None:
            dst.descriptions = tuple(band_names)
                 
def expand_bands_and_reduce(
    dataset: xr.Dataset, 
    features: list
) -> xr.Dataset:
    """
    Expands band and time dimensions into separate variables for each band and month,
    then reduces the dataset by selecting only the specified features.

    Parameters
    ----------
    dataset : xr.Dataset
        Input Xarray dataset containing band_data variable with (band, time, y, x) dimensions.
    features : list
        List of feature names to retain after expansion and reduction.

    Returns
    -------
    xr.Dataset
        Dataset with expanded band-time variables, retaining only selected features.

    Notes
    -----
    - The function creates new variables following the naming convention:
      band_{band_idx+1}_month_{time_idx+1}
    """


    new_vars = {}

    for time_idx in range(len(dataset.time)):
        for band_idx in range(len(dataset.band)):
            var_name = f"band_{band_idx+1}_month_{time_idx+1}"
            # Select the specific band and time slice and squeeze to remove extra dims
            new_vars[var_name] = dataset["band_data"].isel(time=time_idx, band=band_idx)

    # Assign new_vars to the dataset
    dataset = dataset.assign(**new_vars).drop_vars(["band_data", "spatial_ref", "tile"])
    dataset = dataset.drop_dims(["band","time"])
    
    # Filter variables by features list, preserving x and y coordinates
    selected_vars = [var for var in dataset.data_vars if var in features]
    
    dataset = dataset[selected_vars]
    
    return dataset.squeeze()

def map_lc_codes_to_rgba(image_array: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Converts a 2D land cover classification array into a 4-channel RGBA image using a color mapping.

    Parameters
    ----------
    image_array : np.ndarray
        2D array of shape (H, W) with integer class codes.
    color_map : dict
        Mapping from LC_code to normalized RGBA values.

    Returns
    -------
    np.ndarray
        RGBA image array of shape (H, W, 4) with values in [0, 1].
    """
    # Output image with shape (H, W, 4)
    h, w = image_array.shape
    rgba_image = np.zeros((h, w, 4), dtype=np.float32)

    # Assign RGBA values per class
    for lc_code, rgba in color_map.items():
        mask = (image_array == lc_code)
        rgba_image[mask] = rgba
    
    return rgba_image

def load_map_with_probs_and_plot(
    lc_map_path: str,
    labels: list,
    labels_df: pd.DataFrame,
    title: str = "",
    images_path: str = None
) -> np.ndarray:
    """
    Loads a multi-band classification map (with class labels and probabilities),
    extracts the label band, applies RGBA mapping and displays the resulting image.

    Parameters
    ----------
    lc_map_path : str
        Path to the GeoTIFF classification map.
    labels : list
        List of band names, including 'labels' and optional class probabilities.
    labels_df : pd.DataFrame
        DataFrame mapping LC_code to RGBA values.
    title : str, optional
        Title for the plotted map.
    images_path : str, optional
        If provided, saves the plot to this directory.

    Returns
    -------
    np.ndarray
        2D array of predicted land cover class labels.
    """

    lc_map = rioxarray.open_rasterio(lc_map_path)
    lc_map = lc_map.assign_coords(band=labels)
    lc_map = lc_map.sel(band="labels").squeeze(drop=True)
    print(f"\n\nMap at : {lc_map_path}\n\n", lc_map)

    # Color mapping
    color_map = {
        row.LC_code: np.array([row.R, row.G, row.B, row.A]) / 255.0
        for _, row in labels_df.iterrows()
    }
    rgba_image = map_lc_codes_to_rgba(lc_map.values, color_map)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(rgba_image)
    plt.title(title)
    plt.axis("off")
    if images_path:
        plt.savefig(f"{images_path}/{os.path.basename(lc_map_path)[:-4]}.png", bbox_inches='tight')
    plt.show()

    return lc_map.values

def load_map_and_plot(
    lc_map_path: str,
    labels_df: pd.DataFrame,
    title: str = "",
    images_path: str = None
) -> np.ndarray:
    """
    Loads a single-band land cover classification map, applies RGBA color mapping
    and visualizes the result.

    Parameters
    ----------
    lc_map_path : str
        Path to the GeoTIFF file containing label predictions.
    labels_df : pd.DataFrame
        DataFrame mapping LC_code to RGBA values.
    title : str, optional
        Title for the plotted map.
    images_path : str, optional
        If provided, saves the plot to this directory.

    Returns
    -------
    np.ndarray
        2D array of predicted land cover class labels.
    """
    lc_map = rioxarray.open_rasterio(lc_map_path).isel(band = -1).squeeze(drop=True)
    print(f"\n\nMap at : {lc_map_path}\n\n", lc_map)

    # Color mapping
    color_map = {
        row.LC_code: np.array([row.R, row.G, row.B, row.A]) / 255.0
        for _, row in labels_df.iterrows()
    }
    rgba_image = map_lc_codes_to_rgba(lc_map.values, color_map)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(rgba_image)
    plt.title(title)
    plt.axis("off")
    if images_path:
        plt.savefig(f"{images_path}/{os.path.basename(lc_map_path)[:-4]}.png", bbox_inches='tight')
    plt.show()

    return lc_map.values
