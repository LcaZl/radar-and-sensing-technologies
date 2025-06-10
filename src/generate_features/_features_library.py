import numpy as np
import xarray as xr
from scipy.ndimage import sobel
import sys

EPSILON = 1e-40

def band_preparation(band):
    
    return band / 10000

def get_NDVI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Vegetation Index (NDVI), an indicator of 
    vegetation health and biomass. Higher values indicate healthier, denser vegetation.

    NDVI is a standard index for assessing plant vigor and is closely related to GNDVI, 
    though NDVI is more sensitive to canopy structure, while GNDVI is more sensitive 
    to chlorophyll concentration.

    Formula: (NIR - Red) / (NIR + Red + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR)
        - Band 1: Red

    Returns
    -------
    xr.DataArray
        NDVI values with original spatial and temporal dimensions.
    """

    nir = band_preparation(data.sel(band=4))  # NIR
    red = band_preparation(data.sel(band=1))  # Red
    ndvi = (nir - red) / (nir + red + EPSILON)
    ndvi.name = 'NDVI'
    return ndvi

def get_GNDVI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Green Normalized Difference Vegetation Index (GNDVI), which is 
    more sensitive to chlorophyll concentration than NDVI and useful for detecting 
    plant stress earlier.

    Formula: (NIR - Green) / (NIR + Green + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR)
        - Band 2: Green

    Returns
    -------
    xr.DataArray
        GNDVI values with original spatial and temporal dimensions.
    """

    nir = band_preparation(data.sel(band=4))    # NIR (B08)
    green = band_preparation(data.sel(band=2))  # Green (B03)
    gndvi = (nir - green) / (nir + green + EPSILON)
    gndvi.name = 'GNDVI'
    return gndvi

def get_NDVI705(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Red-edge NDVI (NDVI705), enhancing sensitivity to chlorophyll content, 
    especially in cases where traditional NDVI saturates (dense vegetation).

    NDVI705 is complementary to NDVI and GNDVI, providing better insights for early 
    plant stress detection and subtle chlorophyll variations.

    Formula: (NIR - Red-edge) / (NIR + Red-edge + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR)
        - Band 5: Red-edge

    Returns
    -------
    xr.DataArray
        NDVI705 values with original spatial and temporal dimensions.
    """

    nir = band_preparation(data.sel(band=4))         # NIR (B08)
    red_edge = band_preparation(data.sel(band=5))    # Red-edge (B05)
    ndvi705 = (nir - red_edge) / (nir + red_edge + EPSILON)
    ndvi705.name = 'NDVI705'
    return ndvi705

def get_NDYI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Yellow Index (NDYI), designed to highlight 
    areas with yellowing vegetation, which can indicate vegetation degradation.

    While NDVI, NDVI705 and GNDVI focus on healthy vegetation and chlorophyll content, 
    NDYI is particularly useful for detecting early signs of vegetation degradation.

    Formula: (Green - Blue) / (Green + Blue + EPSILON)

    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndyi/

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 2: Green
        - Band 3: Blue

    Returns
    -------
    xr.DataArray
        NDYI values with original spatial and temporal dimensions.
    """

    blue = band_preparation(data.sel(band=3))  # Blue (B02)
    green = band_preparation(data.sel(band=2))  # Green (B03)
    ndyi = (green - blue) / (green + blue + EPSILON)
    ndyi.name = 'NDYI'
    return ndyi

def get_EVI2(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Enhanced Vegetation Index 2 (EVI2), a simplified version of EVI 
    that omits the blue band, making it more robust in areas with atmospheric distortions 
    and cloud cover. EVI2 is effective in high biomass regions and is less sensitive 
    to soil background than NDVI.

    This implementation follows the Sentinel Hub formula:
    EVI2 = 2.5 * (NIR - Red) / (NIR + Red + 1 + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR, B08)
        - Band 1: Red (B04)

    Returns
    -------
    xr.DataArray
        EVI2 values with original spatial and temporal dimensions.
    """
    nir = band_preparation(data.sel(band=4))    # NIR (B08)
    red = band_preparation(data.sel(band=1))    # Red (B04)
    evi2 = 2.4 * (nir - red) / (nir + red + 1 + EPSILON)
    evi2.name = 'EVI2'
    return evi2



def get_GLI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Green Leaf Index (GLI), emphasizing the green band to enhance 
    detection of healthy green vegetation.

    GLI provides a ratio for distinguishing green vegetation 
    from soil and other land cover types.

    Formula: (2 * Green - Red - Blue) / (2 * Green + Red + Blue + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 2: Green
        - Band 1: Red
        - Band 3: Blue

    Returns
    -------
    xr.DataArray
        GLI values with original spatial and temporal dimensions.
    """

    green = band_preparation(data.sel(band=2))  # Green (B03)
    red = band_preparation(data.sel(band=1))    # Red (B04)
    blue = band_preparation(data.sel(band=3))   # Blue (B02)
    gli = (2 * green - red - blue) / (2 * green + red + blue + EPSILON)
    gli.name = 'GLI'
    return gli

def get_SAVI(data: xr.DataArray, L: float = 0.5) -> xr.DataArray:
    """
    Computes the Soil-Adjusted Vegetation Index (SAVI), which modifies NDVI by 
    introducing a correction factor (L) to minimize the influence of soil brightness. 
    This makes it especially effective in areas with sparse or mixed vegetation.

    SAVI behaves similarly to NDVI, but is preferred in low-density vegetation scenarios.

    Formula: SAVI = ((NIR - Red) / (NIR + Red + L + EPSILON)) * (1 + L)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR)
        - Band 1: Red
    L : float, optional
        Soil brightness correction factor. Common values:
        - 0.0 for high vegetation cover
        - 0.5 for intermediate cover (default)
        - 1.0 for very low cover

    Returns
    -------
    xr.DataArray
        SAVI values with original spatial and temporal dimensions.
    """


    nir = band_preparation(data.sel(band=4))
    red = band_preparation(data.sel(band=1))
    savi = ((nir - red) / (nir + red + L + EPSILON)) * (1 + L)
    savi.name = 'SAVI'
    return savi

def get_NDMI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Moisture Index (NDMI), used to assess 
    vegetation water content and detect moisture stress in crops or natural vegetation.

    NDMI is complementary to NDVI but is sensitive to water content rather than chlorophyll.

    Formula: NDMI = (NIR - SWIR1) / (NIR + SWIR1 + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 4: Near-Infrared (NIR, B08)
        - Band 9: Short-Wave Infrared 1 (SWIR1, B11)

    Returns
    -------
    xr.DataArray
        NDMI values with original spatial and temporal dimensions.
    """


    nir = band_preparation(data.sel(band=4))
    swir1 = band_preparation(data.sel(band=9))  # SWIR1
    ndmi = (nir - swir1) / (nir + swir1 + EPSILON)
    ndmi.name = 'NDMI'
    return ndmi

def get_NDLI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Lignin Index (NDLI), which estimates lignin 
    content to distinguish woody vegetation (high lignin) from leafy vegetation (low lignin).

    This index uses the pseudo-absorbance transformation log(1/R) in the SWIR bands,
    based on their differential absorption of lignin.

    Formula: (log(1 / SWIR2) - log(1 / SWIR1)) / (log(1 / SWIR2) + log(1 / SWIR1) + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 9 : SWIR1 (B11)
        - Band 10: SWIR2 (B12)

    Returns
    -------
    xr.DataArray
        NDLI values with original spatial and temporal dimensions.
    """

    swir1 = data.sel(band=9)
    swir2 = data.sel(band=10)

    swir1 = np.clip(swir1, EPSILON, np.inf)
    swir2 = np.clip(swir2, EPSILON, np.inf)

    log_swir1 = np.log(swir1)
    log_swir2 = np.log(swir2)

    numerator = log_swir2 - log_swir1
    denominator = log_swir2 + log_swir1 + EPSILON
    ndli = numerator / denominator

    ndli.name = 'NDLI'
    return ndli



def get_NDWI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Water Index (NDWI), used to enhance the detection 
    of water bodies by comparing reflectance in green and near-infrared bands.

    NDWI highlights open water features, where water absorbs NIR and reflects green light.

    Formula: (Green - NIR) / (Green + NIR + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 2: Green (B03)
        - Band 4: Near-Infrared (NIR, B08)

    Returns
    -------
    xr.DataArray
        NDWI values with original spatial and temporal dimensions.
    """
    
    green = band_preparation(data.sel(band=2)) 
    nir = band_preparation(data.sel(band=4))
    ndwi = (green - nir) / (green + nir + EPSILON)
    ndwi.name = 'NDWI'
    return ndwi

def get_NDBI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Built-up Index (NDBI), which enhances 
    the detection of built-up and urbanized areas.

    NDBI complements indices like MNDWI by focusing on areas with high reflectance 
    in SWIR and lower reflectance in NIR, common in man-made structures.

    Formula: (SWIR1 - NIR) / (SWIR1 + NIR + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 9: Short-Wave Infrared 1 (SWIR1, B11)
        - Band 4: Near-Infrared (NIR, B08)

    Returns
    -------
    xr.DataArray
        NDBI values with original spatial and temporal dimensions.
    """

    swir = band_preparation(data.sel(band=9)) # SWIR1
    nir = band_preparation(data.sel(band=4)) # NIR
    ndbi = (swir - nir) / (swir + nir + EPSILON)
    ndbi.name = 'NDBI'
    return ndbi

def get_NDSI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Normalized Difference Snow Index (NDSI), used to detect snow and ice 
    while distinguishing them from clouds and bare soil.

    This implementation follows the Sentinel Hub script, using green and SWIR1 bands.

    Formula: (Green - SWIR1) / (Green + SWIR1 + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Data array with bands where:
        - Band 2 : Green (B03)
        - Band 9 : Short-Wave Infrared 1 (SWIR1, B11)

    Returns
    -------
    xr.DataArray
        NDSI values with original spatial and temporal dimensions.
    """

    green = band_preparation(data.sel(band=2))
    swir1 = band_preparation(data.sel(band=9))
    ndsi = (green - swir1) / (green + swir1 + EPSILON)
    ndsi.name = 'NDSI'
    return ndsi

def get_BSI(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the Bare Soil Index (BSI), a spectral index designed to identify 
    bare or sparsely vegetated soil surfaces, distinguishing them from vegetation 
    and built-up areas.

    This implementation follows the standard formula using SWIR1, Red, NIR, and Blue bands,
    as described in the Sentinel Hub custom script library.

    Formula:
        BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue) + EPSILON)

    Parameters
    ----------
    data : xr.DataArray
        Multiband reflectance data with bands:
        - Band 9 : SWIR1 (B11)
        - Band 1 : Red (B04)
        - Band 4 : NIR (B08)
        - Band 3 : Blue (B02)

    Returns
    -------
    xr.DataArray
        The BSI index as a 2D DataArray.
    """
    swir = band_preparation(data.sel(band=9))     # SWIR1 (B11)
    red = band_preparation(data.sel(band=1))      # Red (B04)
    nir = band_preparation(data.sel(band=4))      # NIR (B08)
    blue = band_preparation(data.sel(band=3))     # Blue (B02)

    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + EPSILON)
    bsi.name = 'BSI'
    return bsi


def get_sobel(ndvi: xr.DataArray) -> xr.DataArray:
    """
    Applies the Sobel edge detection filter on an NDVI image.

    Computes the gradient magnitude using the Sobel operator in both horizontal 
    and vertical directions. This enhances edges and spatial transitions, useful 
    for detecting boundaries in vegetation cover.

    Parameters
    ----------
    ndvi : xr.DataArray
        A 2D NDVI array with dimensions [y, x].

    Returns
    -------
    xr.DataArray
        A 2D array representing edge intensity, highlighting NDVI boundaries.
    """

    sobel_x = sobel(ndvi, axis=0)
    sobel_y = sobel(ndvi, axis=1)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    #sobel_magnitude *= 255.0 / np.max(sobel_magnitude)

    return xr.DataArray(
        sobel_magnitude,
        coords=ndvi.coords,
        dims=ndvi.dims,
        name='Sobel'
    )
