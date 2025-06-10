import numpy as np
import xarray as xr
import cv2
import logging

def restore_cloud_shadow_xr(
    single_band_composites: xr.DataArray
) -> xr.DataArray:
    """
    Restores cloud/shadow-affected regions in a single-band composites dataset 
    by interpolating missing values (represented as zeros) over the time dimension.

    Parameters
    ----------
    single_band_composites : xr.DataArray
        An xarray DataArray of shape [time, y, x], where missing values 
        due to clouds or shadows are represented as zeros.

    Returns
    -------
    xr.DataArray
        The restored DataArray with missing values replaced using temporal interpolation.

    Notes
    -----
    - The dataset is sorted by the time dimension before interpolation.
    - If a missing value is at the first time step, it is replaced by the next valid value.
    - If a missing value is at the last time step, it is replaced by the previous valid value.
    - If a missing value is in the middle, it is replaced by the average of the 
      previous and next valid values.
    """


    zero_mask = (single_band_composites == 0)  | (single_band_composites == np.nan)

    single_band_composites = single_band_composites.sortby("time")
    
    # Shift to get previous and next valid time slices
    prev_valid = single_band_composites.shift(time=1)  # Shift down (previous time)
    next_valid = single_band_composites.shift(time=-1) # Shift up (next time)

    # Case 1: Missing at the first time step -> use next valid value
    restored_values = xr.where(prev_valid.isnull(), next_valid, np.nan)

    # Case 2: Missing at the last time step -> use previous valid value
    restored_values = xr.where(next_valid.isnull(), prev_valid, restored_values)

    # Case 3: Missing in the middle take -> the average of previous and next valid values
    restored_values = xr.where(restored_values.isnull(), (prev_valid + next_valid) / 2, restored_values)

    # Replace missing pixels (zeros) with the computed values
    restored = xr.where(zero_mask, restored_values, single_band_composites)

    return restored


def shadow_adjustment(
    image: np.ndarray, 
    slope: np.ndarray
) -> np.ndarray:
    """
    Adjusts shadowed areas in a multispectral image array using slope and spectral band information
    to identify and correct shadows.

    Algorithm
    ---------
    1. Extract Bands:
       - The function uses the blue, NIR (near-infrared), and SWIR (shortwave infrared) bands from the input image.
    
    2. Compute Cloud Shadow Index (CSI):
       - Calculate a Cloud Shadow Index ('csi') as the average of the NIR and SWIR bands.
       - Compute two thresholds:
         - 't1' for the CSI, based on its minimum and mean values.
         - 't2' for the blue band, also based on its minimum and mean values.

    3. Identify Shadow Mask:
       - Create a binary mask ('mask') where pixels meet both CSI and blue band thresholds (likely shadow areas).
       - Dilate the shadow mask using a morphological ellipse to slightly expand shadow regions.

    4. Remove Water Areas:
       - Apply the slope data to filter out regions with a slope below a threshold (15° here) to exclude water areas.

    5. Separate Shadowed and Non-Shadowed Pixels:
       - Use the mask to separate shadowed and non-shadowed regions in each band.
       - Calculate the mean and standard deviation for shadowed and non-shadowed areas.

    6. Match Shadows to Non-Shadowed Pixels:
       - For shadowed pixels, compute an adjustment parameter to match their intensity distribution to that of non-shadowed pixels.
       - Adjust the values of shadowed pixels and combine with non-shadowed pixels to produce a "shadow-corrected" image.

    7. Inpaint Small Gaps:
       - Create a "ring" mask around shadowed regions, dilate and erode the shadow mask to form a boundary.
       - Fill small gaps along shadow edges using inpainting.

    Parameters
    ----------
    image : np.ndarray
        The multispectral image with shape (bands, height, width) to apply shadow adjustment.
    slope : np.ndarray
        The slope data for the area, used to filter water regions based on slope values.

    Returns
    -------
    np.ndarray
        The adjusted multispectral image with shadow corrections applied.

    """

    logger = logging.getLogger("shadow-adj")
    blue = image[3]
    nir = image[4] # or 7
    swir = image[9] # or 9
    nb = image.shape[0]

    csi = ((nir + swir) / 2)   
    t1 = (np.nanmin(csi) + 0.5 * (np.nanmean(csi) - np.nanmin(csi)))
    t2 = (np.nanmin(blue) + 0.25 * (np.nanmean(blue) - np.nanmin(blue)) )
    csi_th = csi <= t1
    blue_th = blue <= t2
    mask = (csi_th*blue_th).astype(np.uint8)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # define a disk element
    mask_dil = cv2.dilate(mask,se,iterations = 1)

    # threshold for water removal check
    mask_dil = slope * mask_dil
    mask_dil =  mask_dil > 15  # threshold for water removal 
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # define a disk element
    mask_dil = cv2.dilate(mask_dil.astype(np.uint8),se,iterations = 1)

    mask_dil_inv = mask_dil ^ 1

    se= np.ones((3,3))
    mask_dil_dil = cv2.dilate(mask_dil,se,iterations = 1)
    mask_dil_ero = cv2.erode(mask_dil,se,iterations = 1)
    mask_ring = mask_dil_dil - mask_dil_ero
    
    out = np.zeros_like(image, dtype=np.uint16)
    
    for b in range(nb):
        
        band = image[b]
        
        band_sh = np.where(mask_dil, band, np.nan).astype(np.float32)      
        band_nosh = np.where(mask_dil_inv, band, np.nan).astype(np.float32)

        mean_sh = np.nanmean(band_sh)
        mean_nosh = np.nanmean(band_nosh)
        std_sh = np.nanstd(band_sh,  ddof=1)
        std_nosh = np.nanstd(band_nosh,  ddof=1)

        if (
            np.isnan(mean_sh) or np.isnan(mean_nosh) or 
            std_sh == 0 or std_nosh == 0
        ):
            # no shadows or invalid stats => keep original band
            logger.info(f"Band {b} shadow correction not possible.")
            band_corrected = band
            
        else:
            # match histograms
            par_b = std_nosh / std_sh
            par_a = mean_nosh - par_b * mean_sh
            band_sh_matched = (band_sh * par_b) + par_a

            # re-combine
            band_corrected = np.where(mask_dil, band_sh_matched, band_nosh)
      
        
        band_inpaint = cv2.inpaint(band_corrected.astype(np.float32), mask_ring, 3, cv2.INPAINT_TELEA)
        out[b] = np.rint(band_inpaint).clip(0, 65535).astype(np.uint16)
        del band_sh, band_nosh, band_sh_matched, band_corrected, band_inpaint
    del image
    del slope
    del band   
    del blue
    del nir
    del swir

    return out[np.newaxis, ...]


