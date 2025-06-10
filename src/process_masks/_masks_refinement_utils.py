import logging
import xarray as xr
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import medfilt2d
import dask.array as da

########################################################
#       Background generation & Masks refinement       #
########################################################

def apply_mask_and_prepare(
    image: np.ndarray, 
    cloud_mask: np.ndarray, 
    shadow_mask: np.ndarray
) -> np.ndarray:
    """
    Applies cloud and shadow masks to an image, replaces masked pixels with NaN 
    and reshapes the result into a 1D array.

    Parameters
    ----------
    image : np.ndarray
        2D image array (y, x).
    cloud_mask : np.ndarray
        Binary mask for clouds (1 for cloud, 0 otherwise).
    shadow_mask : np.ndarray
        Binary mask for shadows (1 for shadow, 0 otherwise).

    Returns
    -------
    np.ndarray
        The masked and reshaped image as a 1D array with zeros replaced by NaN.
    """

    # Step 1: Apply masks (1 for cloud/shadow -> masked)
    combined_mask = cloud_mask + shadow_mask
    not_cloud_shadow_mask = combined_mask ^ 1

    # Step 2: Apply mask and replace 0s with NaN
    masked_image = np.multiply(image, not_cloud_shadow_mask).astype(np.float32)
    
    masked_image[masked_image == 0] = np.nan

    #masked_image = np.where(cloud_mask or shadow_mask, np.nan, image).astype(np.float32)
    
    # Step 3: Reshape to (pixels,)
    return masked_image.reshape(-1)

def band_preparation(slice_2d: np.ndarray) -> np.ndarray:
    """
    Clips the input 2D band slice to the 0.001 - 0.999 quantile range to suppress extreme outliers.

    Parameters
    ----------
    slice_2d : np.ndarray
        A 2D NumPy array representing a single spectral band image (e.g., shape [y, x]).

    Returns
    -------
    np.ndarray
        The clipped 2D array with values limited to the [0.001, 0.999] quantile range.
    """
    maxVal = np.quantile(slice_2d, 0.999)
    minVal = np.quantile(slice_2d, 0.001)
    return np.clip(slice_2d, minVal, maxVal)

def compute_custom_nanpercentile(
    blue_band: np.ndarray, 
    cloud_mask: np.ndarray, 
    shadow_mask: np.ndarray, 
    q: float
) -> np.ndarray:
    """
    Computes a custom nanpercentile over time for a blue band, applying cloud and 
    shadow masks individually to each time slice.

    Parameters
    ----------
    blue_band : np.ndarray
        3D array of blue band data with shape (y, x, time).
    cloud_mask : np.ndarray
        3D binary cloud mask array (1 for cloud, 0 otherwise).
    shadow_mask : np.ndarray
        3D binary shadow mask array (1 for shadow, 0 otherwise).
    q : float
        Quantile value to compute (e.g., 0.25 for the 25th percentile).

    Returns
    -------
    np.ndarray
        2D array (y, x) with computed percentile values, where masked pixels 
        are excluded from the computation.

    """
    logger = logging.getLogger("bg-chunk-processing")
    
    logger.info(f"Processing patch with shape: {blue_band.shape}")  # (y, x, time)

    y_size, x_size, time_size = blue_band.shape
    pixel_count = y_size * x_size
        
    clipped = np.empty_like(blue_band)
    for t in range(blue_band.shape[2]):
        clipped[:, :, t] = band_preparation(blue_band[:, :, t])
        
    blue_band = clipped
    
    # Prepare the time series (like loadTimeSeries2D)
    ts_2D = np.zeros((pixel_count, time_size), dtype=np.float32)

    for t in range(time_size):
        ts_2D[:, t] = apply_mask_and_prepare(
            blue_band[..., t], cloud_mask[..., t], shadow_mask[..., t]
        )

    # Step 4: Compute mask for valid pixels and group counts
    mask = (ts_2D >= np.nanmin(ts_2D)).astype(int)
    count = mask.sum(axis=1)
    groups = np.unique(count)
    groups = groups[groups > 0]

    # Step 5: Process groups and compute percentile
    temp = np.zeros((pixel_count,), dtype=np.float32)

    for g in groups:
        pos = np.where(count == g)
        values = ts_2D[pos]
        values = np.nan_to_num(values, nan=0)
        values = np.sort(values, axis=1)
        values = values[:, -g:]  # Keep last 'g' valid values

        # Compute percentile using 'midpoint' interpolation (as in old code)
        temp[pos] = np.percentile(values, q * 100, axis=1, interpolation="midpoint")

    # Reshape back to (y, x)
    return temp.reshape((y_size, x_size))

def generate_seasonal_backgrounds(
    dataset: xr.Dataset, 
    quantile: float = 0.25
) -> xr.DataArray:
    """
    Generates seasonal background images by computing a specified quantile across 
    the time dimension, excluding regions affected by clouds and shadows.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing:
        - "data": Multi-band composite data with a 'band' dimension.
        - "masks": Mask dataset indicating clouds and shadows.
        - "season_id": Identifier for each season.
    quantile : float, optional
        The quantile value to compute (0.25).

    Returns
    -------
    xr.DataArray
        Concatenated seasonal background images with the specified quantile, 
        indexed by 'season_id'.

    """

    logger = logging.getLogger("generate-backgrounds")
    # Select blue band and cloud and shadow masks
    blue_band = dataset.data.sel(band="B2").astype(np.float32)
    cloud_mask = dataset.masks.sel(mask_type="cloud")
    shadow_mask = dataset.masks.sel(mask_type="shadow")

    blue_band = blue_band.chunk({"time": -1, "x": 2196, "y": 2196})
    cloud_mask = cloud_mask.chunk({"time": -1, "x": 2196, "y": 2196})
    shadow_mask = shadow_mask.chunk({"time": -1, "x": 2196, "y": 2196})

    logger.info(f"Data Shapes - Blue Band: {blue_band.shape}, Cloud Mask: {cloud_mask.shape}, Shadow Mask: {shadow_mask.shape}")

    backgrounds_list = []
    season_ids = np.unique(dataset.season_id.values)

    for season in season_ids:
        logger.info(f"Processing season: {season}")

        season_indices = dataset.season_id == season
        season_blue = blue_band.sel(time=dataset.time[season_indices])
        season_cloud = cloud_mask.sel(time=dataset.time[season_indices])
        season_shadow = shadow_mask.sel(time=dataset.time[season_indices])

        background = xr.apply_ufunc(
            compute_custom_nanpercentile,
            season_blue,
            season_cloud,
            season_shadow,
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[[]],
            dask="parallelized",
            kwargs={"q": quantile},
            output_dtypes=[np.float32]
        )

        background = background.assign_coords({
            "x": dataset.x,
            "y": dataset.y,
            "season_id": season
        })

        backgrounds_list.append(background)

    # Concatenate all seasonal backgrounds
    backgrounds = xr.concat(backgrounds_list, dim="season_id")

    logger.info("Generated background images successfully!")
    return backgrounds

def refine_cloud_mask(blue_band : np.ndarray, cloud_mask : np.ndarray, background_img : np.ndarray, 
                      cloud_coverage_threshold : float) -> np.ndarray:
    """
    Refines the Sen2Cor cloud mask by comparing the current blue band image to a precomputed seasonal background image. 
    The refinement process enhances the detection of clouds by identifying regions where the blue band deviates significantly 
    from the cloud-free background. If the cloud coverage exceeds a threshold, KMeans clustering is applied to further refine 
    the cloud mask. The process can be summarized as follow:

    1. The absolute difference between the blue band and the background is reshaped into a one-dimensional array for clustering.
    2. KMeans clustering is applied to this array, segmenting the image into clusters of similar pixel intensities.
    3. The cluster means are calculated, and their distances from the mean reflectance of cloud pixels in the blue band are measured.
    4. The cluster with the mean value closest to the cloud region (based on the original cloud mask) is identified.
    5. The cloud mask is refined by selecting pixels from the closest cluster, which are then added to the existing cloud mask.

    If the cloud coverage is below the specified threshold, the original cloud mask 
    is returned unchanged.
    
    Parameters
    ----------
    blue_band : np.ndarray
        The blue band (B2) of the satellite image, used for cloud detection.
    cloud_mask : np.ndarray
        The initial cloud mask generated by Sen2Cor, which will be refined.
    background_img : np.ndarray
        The precomputed seasonal background image for the corresponding scene.
    cloud_coverage_threshold : float
        The percentage of cloud cover required to trigger mask refinement.

    Returns
    -------
    np.ndarray
        The refined cloud mask where cloud pixels are marked as True.
    """ 
    logger = logging.getLogger("cloud-mask-refinement")
    blue_band = blue_band.squeeze()
    
    
    cloud_mask = cloud_mask.squeeze()
    background_img = background_img.squeeze()

    if not blue_band.shape == cloud_mask.shape:
        raise ValueError(f"Input shapes must match", blue_band.shape, cloud_mask.shape, background_img.shape)

    # Calculate cloud coverage percentage
    num = np.unique(cloud_mask, return_counts=True)[1]
    cloud_percentage = num[1] / (num[0] + num[1])

    # Skip refinement if cloud coverage is below the threshold
    logger.info(f"Cloud coverage percentage: {cloud_percentage} (Status: {cloud_percentage > cloud_coverage_threshold})")
    if cloud_percentage <= cloud_coverage_threshold:
        return cloud_mask
    
    # Compute absolute difference between the blue band and the background image
    diff = np.abs(blue_band - background_img) 

    # Flatten for clustering
    diff_flat = diff.reshape(-1, 1)
    diff_flat = np.nan_to_num(diff_flat, nan=0)

    # Perform KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(diff_flat)
    cluster_labels = kmeans.labels_.reshape(diff.shape)

    # Compute the mean of each cluster and its distance to the mean cloud reflectance
    mean_cloud_reflectance = (cloud_mask * blue_band).astype(float)
    mean_cloud_reflectance[mean_cloud_reflectance == 0.0] = np.nan
    mean_cloud_reflectance = np.nanmean(mean_cloud_reflectance)

    means = []
    distances = []
    for c in range(3):
        cluster_mask = (cluster_labels == c)
        cluster_values = diff * cluster_mask
        cluster_values = np.where(cluster_values == 0, np.nan, cluster_values)
        cluster_mean = np.nanmean(cluster_values)
        means.append(cluster_mean)
        distances.append(abs(mean_cloud_reflectance - cluster_mean))

    # Identify the cluster closest to the cloud reflectance
    closest_cluster_idx = np.argsort(distances)[0]
    closest_cluster_mask = (cluster_labels == closest_cluster_idx)

    refined_mask = closest_cluster_mask | cloud_mask
    refined_mask = np.where(refined_mask > 1, 1, refined_mask)
    logger.info("Cloud mask refinement complete.")

    return refined_mask

def refine_shadow_mask(blue_band : np.ndarray, nir_band : np.ndarray, swir_band : np.ndarray, cloud_mask : np.ndarray, shadow_mask : np.ndarray, 
                       cloud_coverage_threshold : float, image_brightness_threshold : int) -> np.ndarray:
    """
    Refines the Sen2Cor shadow mask by analyzing the blue, NIR and SWIR bands. 
    The process begins by evaluating the cloud coverage and cloud reflectance. If the cloud coverage exceeds the given threshold 
    and the mean cloud reflectance is above the brightness threshold, the function proceeds with shadow detection refinement.

    The refinement process excludes cloud pixels by masking them out and computes a Cloud Shadow Index (CSI) as the mean of the 
    NIR and SWIR bands. Two thresholds are then calculated:
    
    1. A threshold (t1) based on the CSI values, where shadows are expected to have low CSI.
    2. A threshold (t2) based on the blue band values, where shadows have low reflectance.

    These thresholds are applied to detect potential shadow pixels by identifying areas where both CSI and blue band values 
    fall below their respective thresholds. The resulting shadow candidate mask is smoothed using a median filter to reduce noise, 
    and it is combined with the initial shadow mask to produce a refined shadow mask.

    If the cloud coverage or brightness does not meet the specified thresholds, the original shadow mask is returned unchanged.

    Parameters
    ----------
    blue_band : np.ndarray
        The blue band (B2) of the input image, used for detecting shadows.
    nir_band : np.ndarray
        The NIR band (B8) of the input image, used for shadow index calculation.
    swir_band : np.ndarray
        The SWIR band (B11) of the input image, used for shadow index calculation.
    cloud_mask : np.ndarray
        The initial cloud mask generated by Sen2Cor.
    shadow_mask : np.ndarray
        The initial shadow mask generated by Sen2Cor.
    cloud_coverage_threshold : float
        The minimum cloud coverage percentage required to trigger shadow mask refinement.
    image_brightness_threshold : int
        The brightness threshold for cloud reflectance in the blue band.

    Returns
    -------
    np.ndarray
        The refined shadow mask where shadow pixels are marked as True.
    """
    logger = logging.getLogger(f"shadow-mask-refinement")

    blue_band = blue_band.squeeze()
    cloud_mask = cloud_mask.squeeze().astype(bool)
    nir_band = nir_band.squeeze()
    swir_band = swir_band.squeeze()
    shadow_mask = shadow_mask.squeeze()

    if not blue_band.shape == cloud_mask.shape == nir_band.shape == swir_band.shape == shadow_mask.shape:
        logger.error(f"Error: input images to shadow refinement procedure have different shapes")
        logger.error(f"Shapes\nBlue band: {blue_band.shape} - Cloud mask: {cloud_mask.shape} - Nir : {nir_band.shape} - swir: {swir_band.shape} - shadow: {shadow_mask.shape}")
        raise ValueError("Input shapes must match")

    # Compute cloud coverage percentage
    num_cloud = np.sum(cloud_mask)
    num_total = cloud_mask.size
    cloud_percentage = num_cloud / num_total
    
    # Compute mean cloud reflectance in cloud pixels
    mean_cloud_reflectance = np.mean(blue_band, where = cloud_mask)
    
    logger.info(f"Cloud coverage percentage: {cloud_percentage} (Status: {cloud_percentage > cloud_coverage_threshold}) - Mean cloud reflectance: {mean_cloud_reflectance} (Status: {mean_cloud_reflectance > image_brightness_threshold})")
    
    if cloud_percentage > cloud_coverage_threshold and mean_cloud_reflectance > image_brightness_threshold:
        
        # Set cloud pixels to NaN to exclude them from calculations
        blue_slice = np.where(cloud_mask, np.nan, blue_band)
        nir_slice = np.where(cloud_mask, np.nan, nir_band)
        swir_slice = np.where(cloud_mask, np.nan, swir_band)
        
        # Compute CSI
        csi = (nir_slice + swir_slice) / 2

        # Compute thresholds t1 and t2
        csi_min = np.nanmin(csi)
        csi_mean = np.nanmean(csi)
        t1 = csi_min + 0.5 * (csi_mean - csi_min)
        #logger.info(f"Threshold T1: {t1} - Min. CSI: {csi_min} - Mean CSI: {csi_mean}")

        blue_min = np.nanmin(blue_slice)
        blue_mean = np.nanmean(blue_slice)
        t2 = blue_min + 0.25 * (blue_mean - blue_min)
        #logger.info(f"Threshold T2: {t2} - Min. BLUE: {csi_min} - Mean BLUE: {csi_mean}")

        # Apply thresholds
        csi_th = csi <= t1
        blue_th = blue_slice <= t2

        # Combine thresholds to get the shadow mask
        mask = (csi_th & blue_th).astype(np.uint8)

        # Apply median filter
        mask = medfilt2d(mask, kernel_size=3)

        # Add to existing shadow mask
        refined_mask = mask | shadow_mask 
    else:
        
        logger.info(f"Refinement not possible. New mask is equal to sen2cor shadow maks.")
        refined_mask = shadow_mask

    refined_mask = np.where(refined_mask > 1, 1, refined_mask)

    logger.info("Shadow mask refinement complete.")
    return refined_mask
