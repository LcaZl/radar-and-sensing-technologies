import numpy as np
import xarray as xr
import logging
from scipy.interpolate import RectBivariateSpline
from skimage.feature import graycomatrix, graycoprops
from numpy.lib.stride_tricks import as_strided as ast
from joblib import Parallel, delayed


def computeGLCM(merge : np.ndarray, win_size : int , distances : list[int], angle : list[int], batch_size : int = 10):
    """
    Computes Gray-Level Co-occurrence Matrix (GLCM) features for an input image
    using a sliding window approach.

    Parameters
    ----------
    merge : np.ndarray
        Input image (2D array) for which GLCM features have to be computed.
    win_size : int
        Size of the sliding window (square window of size `win_size x win_size`).
    distances : list[int]
        List of pixel distances for calculating the GLCM.
    angle : list[float]
        List of angles (in radians) for calculating the GLCM.
    batch_size : int, optional
        Number of windows processed simultaneously in parallel with joblib. Default is 10.

    Returns
    -------
    dict
        A dictionary containing six GLCM features for the input image:
        - "contrast": GLCM contrast.
        - "dissimilarity": GLCM dissimilarity.
        - "homogeneity": GLCM homogeneity.
        - "energy": GLCM energy.
        - "correlation": GLCM correlation.
        - "ASM": Angular Second Moment (ASM).
        Each feature is returned as a 2D array resized to the original image dimensions.

    Notes
    -----
    - Zero values in the input image are treated as missing and replaced with NaN in the output.
    - The p_me, sliding_window and related functions are not modified from their structure in previous pipeline.
    """

    logger = logging.getLogger("compute-GLCM")
    win = win_size
    merge = (merge - np.min(merge)) / (np.max(merge) - np.min(merge))
    merge *= 254
    merge += 1
    
    logger.info("GLCM computation info:")
    print(f"Window used                   : {win} x {win}")
    print(f"GLCM Distance                 : {distances}")
    print(f"GLCM Angle                    : {angle}")
    print(f"Batch size for joblib.Parallel: {batch_size}")

    z, ind = sliding_window(merge, (win, win), (win, win))
    Ny, Nx = np.shape(merge)

    z = z.astype(np.uint8)
    logger.info(f"Start computation over {len(z)} windows")

    def process_window(idx):
        return p_me(z[idx], distances, angle)

    logger.info(f"Total tasks: {len(z)}")
    with Parallel(n_jobs=-1, verbose=2, batch_size=batch_size) as parallel:
        results = parallel(
            delayed(process_window)(k) for k in range(len(z))
        )

    cont, diss, homo, eng, corr, ASM = zip(*results)  # Unpack results

    plt_cont = np.reshape(cont, (ind[0], ind[1]))
    plt_diss = np.reshape(diss, (ind[0], ind[1]))
    plt_homo = np.reshape(homo, (ind[0], ind[1]))
    plt_eng = np.reshape(eng, (ind[0], ind[1]))
    plt_corr = np.reshape(corr, (ind[0], ind[1]))
    plt_ASM = np.reshape(ASM, (ind[0], ind[1]))

    del cont, diss, homo, eng, corr, ASM

    contrast = im_resize(plt_cont, Nx, Ny)
    contrast[merge == 0] = np.nan
    dissimilarity = im_resize(plt_diss, Nx, Ny)
    dissimilarity[merge == 0] = np.nan
    homogeneity = im_resize(plt_homo, Nx, Ny)
    homogeneity[merge == 0] = np.nan
    energy = im_resize(plt_eng, Nx, Ny)
    energy[merge == 0] = np.nan
    correlation = im_resize(plt_corr, Nx, Ny)
    correlation[merge == 0] = np.nan
    ASM = im_resize(plt_ASM, Nx, Ny)
    ASM[merge == 0] = np.nan

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "ASM": ASM
    }

def im_resize(
    im: np.ndarray, 
    Nx: int, 
    Ny: int
) -> np.ndarray:
    """
    Resizes a 2D array (image) using bivariate spline interpolation.

    Parameters
    ----------
    im : np.ndarray
        Input 2D array or image to be resized.
    Nx : int
        Target number of columns (width) in the resized image.
    Ny : int
        Target number of rows (height) in the resized image.

    Returns
    -------
    np.ndarray
        Resized 2D array.
    """

    ny, nx = np.shape(im)
    xx = np.linspace(0, nx, Nx)
    yy = np.linspace(0, ny, Ny)

    newKernel = RectBivariateSpline(np.r_[:ny], np.r_[:nx], im)
    return newKernel(yy, xx)

def norm_shape(
    shap: int | tuple
) -> tuple:
    """
    Normalizes input shape to always be expressed as a tuple.

    Parameters
    ----------
    shap : int or tuple
        Input shape as an integer or a tuple of integers.

    Returns
    -------
    tuple
        Normalized shape as a tuple.
    
    Raises
    ------
    TypeError
        If the input is neither an integer nor a tuple of integers.
    """

    try:
        i = int(shap)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shap)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError("shape must be an int, or a tuple of ints")

def p_me(
    data: np.ndarray, 
    distance: int, 
    angle: float
) -> tuple:
    """
    Calculates GLCM (Gray Level Co-occurrence Matrix) properties for a given image window.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array representing the image window.
    distance : int
        Distance between pixel pairs for GLCM calculation.
    angle : float
        Angle between pixel pairs for GLCM calculation.

    Returns
    -------
    tuple
        GLCM properties:
        - contrast
        - dissimilarity
        - homogeneity
        - energy
        - correlation
        - ASM (Angular Second Moment)
    """

    try:
        image = np.copy(data)
        glcm = graycomatrix(image=image, distances=[distance] , angles=[angle], symmetric=True, normed=True)

        cont = graycoprops(glcm, "contrast")
        diss = graycoprops(glcm, "dissimilarity")
        homo = graycoprops(glcm, "homogeneity")
        eng = graycoprops(glcm, "energy")
        corr = graycoprops(glcm, "correlation")
        ASM = graycoprops(glcm, "ASM")
        
        del glcm
        return cont, diss, homo, eng, corr, ASM
    except Exception as e:
        return 0, 0, 0, 0, 0, 0


def sliding_window(
    a: np.ndarray, 
    ws: int | tuple, 
    ss: int | tuple = None, 
    flatten: bool = True
) -> tuple:
    """
    Extracts sliding windows from an n-dimensional numpy array with customizable 
    window size and step size, optionally flattening each window.

    Parameters
    ----------
    a : np.ndarray
        An n-dimensional numpy array from which to extract sliding windows.
    ws : int or tuple
        Size of each window dimension. If an int is provided, the same size is 
        used for all dimensions. If a tuple is provided, each dimension can 
        have a different size.
    ss : int or tuple, optional
        Step size (stride) for sliding the window in each dimension. If not 
        provided, it defaults to ws, resulting in non-overlapping windows.
    flatten : bool, optional
        If True, each window is flattened into a 1D array. If False, the windows 
        retain the same number of dimensions as the input array.

    Returns
    -------
    tuple
        - np.ndarray: Array containing the extracted sliding windows.
        - tuple: The shape of the strided array.

    Raises
    ------
    ValueError
        If the number of dimensions in a.shape, ws and ss do not match.
        If any window size in ws is larger than the corresponding dimension in a.

    Notes
    -----
    - This function supports n-dimensional arrays.
    - If flatten=True, the output array has one more dimension than the window.
    - The source of this implementation is from:
      http://www.johnvinyard.com/blog/?p=268#more-268
    """
    
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        "a.shape, ws and ss must all have the same length. They were %s" % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        "ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s" % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)


    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)


    # the strides tuple will be the array"s strides multiplied by step size, plus
    # the array"s strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.prod(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    
   # dim = filter(lambda i : i != 1,dim) iwona commented!

    return a.reshape(dim), newshape  

