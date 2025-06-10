import os
import logging
import xarray as xr
import numpy as np
from distributed import Client
import rioxarray
from threading import Thread

from ._masks_refinement_utils import (
    refine_cloud_mask,
    refine_shadow_mask,
    generate_seasonal_backgrounds,
)

from scripting import (
    logged_main,
    print_map,
    monitor_memory,
    load_s2,
    preprocess,
    set_bands,
    drop_aux_bands,
    get_scl_mask
)

def shadow_masks_refinement(
    group: xr.Dataset, 
    refined_cloud_mask: xr.DataArray, 
    cloud_coverage_threshold: float, 
    image_brightness_threshold: int
) -> xr.DataArray:
    """
    Wrapper function that uses xr.apply_ufunc for parallelize computation with Dask of the
    shadow masks refinement process.

    Parameters
    ----------
    group : xr.Dataset
        Xarray dataset containing bands and masks.
    refined_cloud_mask : xr.DataArray
        Refined cloud mask used for shadow refinement.
    cloud_coverage_threshold : float
        Cloud coverage percentage required to trigger shadow refinement.
    image_brightness_threshold : int
        Brightness threshold for cloud reflectance in the blue band.

    Returns
    -------
    xr.DataArray
        Refined shadow mask as an Xarray DataArray.
    """
    return xr.apply_ufunc(
        refine_shadow_mask,       
        group.data.sel(band="B2"),             # Blue band (B2)
        group.data.sel(band="B8"),             # NIR band (B8)
        group.data.sel(band="B11"),            # SWIR band (B11)
        refined_cloud_mask,    # Cloud mask
        group.masks.sel(mask_type="shadow"),   # Shadow mask
        kwargs={"cloud_coverage_threshold"   : cloud_coverage_threshold,
                "image_brightness_threshold" : image_brightness_threshold},
        input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x'], ['y', 'x'], ['y', 'x']],
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        keep_attrs=True,
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    
def cloud_masks_refinement(
    group: xr.Dataset, 
    background: xr.DataArray, 
    cloud_coverage_threshold: float
) -> xr.DataArray:
    """
    Wrapper function that uses xr.apply_ufunc for parallelize computation with Dask of the
    cloud masks refinement process.
    
    Parameters
    ----------
    group : xr.Dataset
        Xarray dataset containing blue band and cloud masks.
    background : xr.DataArray
        Background image used for cloud mask refinement.
    cloud_coverage_threshold : float
        Cloud coverage percentage required to trigger cloud mask refinement.

    Returns
    -------
    xr.DataArray
        Refined cloud mask as an Xarray DataArray.
    """
    return xr.apply_ufunc(
        refine_cloud_mask,
        group.data.sel(band="B2"),                            # Blue band (B2) as input
        group.masks.sel(mask_type="cloud"),  
        background,# Cloud mask
        kwargs = {'cloud_coverage_threshold':cloud_coverage_threshold},
        input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x']], 
        output_core_dims=[['y', 'x']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        keep_attrs=True,
        dask_gufunc_kwargs={'allow_rechunk': True}
    )


def masks_refinement(
    sensor : str,
    year : str,
    tile_id : str,
    verbose : bool,
    input_path : str,
    output_path : str,
    resolutions : list[str],
    mask_definitions,
    cloud_coverage_threshold : float,
    image_brightness_threshold : int,
    **kwargs
) -> None:

    performance_file = f"{kwargs['log_dir']}/{kwargs['log_name'][:-4]}.csv"
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()

    with Client(
        processes = False,
        threads_per_worker = 8,
    ) as client:
        
        # Initial setup
        
        logger = logging.getLogger("masks-refinement")
 
        dask_graph_path = f"{output_path}/dask_graph/"
        bg_output_path = f"{output_path}/backgrounds"
        cloud_masks_path = f"{output_path}/cloud_masks"
        shadow_masks_path = f"{output_path}/shadow_masks"
        
        os.makedirs(cloud_masks_path, exist_ok=True)
        os.makedirs(shadow_masks_path, exist_ok=True)
        os.makedirs(bg_output_path, exist_ok=True)
        os.makedirs(dask_graph_path, exist_ok=True)

        logger.info(f"Dask dashboard: {client.dashboard_link}")
        logger.info(f"Shadow & Cloud Masks refinement for tile_id {tile_id} with sensor {sensor} for {year}.")
        
        # Load datasets, set the band name and extract SCL masks
        
        dss = {res: load_s2(input_path, 
                            group=res, 
                            preprocess=preprocess,
                            tile=tile_id,
                            sensor=sensor,
                            year=year) for res in resolutions}

        dss = {res: set_bands(dss[res], only_bands=False) for res in resolutions}
        scls = {res : dss[res].sel(band='SCL') for res in resolutions if res in ["20m", "60m"]} # Isolating SCLs band
        dss, band_names = drop_aux_bands(**dss) # Dropping auxiliary bands
        
        # Interpolate SCL masks to 10m resolutions
        if "10m" in resolutions:
            logger.info(f"Estimating SCL mask for 10m resolution")
            ref = "20m" if "20m" in resolutions else "60m"
            scls["10m"] = scls[ref].interp(
                
                dict(x=dss["10m"].coords["x"], y=dss["10m"].coords["y"]),
                method="nearest",
                kwargs=dict(fill_value="extrapolate"),
            )        

        # Retrieving SCL masks accordingly to requested layers
        
        logger.info(f"Start retrieving masks from SCL band. Masks Requested: {mask_definitions}")

        scl_masks = xr.concat([
            xr.concat(
                [
                    get_scl_mask(scls[resolutions[0]].sel(time=t), scl_values)
                    .expand_dims({"mask_type": [mask_type]})
                    .astype(bool)
                    for mask_type, scl_values in mask_definitions.items()
                ], 
                dim="mask_type"
            )
            for t in scls[resolutions[0]].time.values
        ], dim="time")

        # Interpolate datasets with different resolutions to a single one with 10m resolution.
        
        logger.info(f"Interpolating only necessary bands from {resolutions[1:]} to {resolutions[0]}")
        dss_up: dict[str, xr.Dataset] = dict()

        # Keep the 10m data with bands B2 and B8 as is
        dss_up[resolutions[0]] = dss[resolutions[0]].sel(band=["B2", "B8"])

        # Interpolate only B11 from the 20m resolution dataset
        if "B11" in dss[resolutions[1]].band:
            dss_up["20m"] = (
                dss["20m"]
                .sel(band="B11")
                .interp(
                    dict(
                        x=dss_up[resolutions[0]].coords["x"],
                        y=dss_up[resolutions[0]].coords["y"],
                    ),
                    method="nearest",
                    kwargs=dict(fill_value="extrapolate"),
                )
                .astype(np.uint16)
            )

        # Concatenate the 10m data (B2, B8) with the interpolated B11
        ds = xr.concat([dss_up[resolutions[0]], dss_up["20m"]], dim="band")
        
        # Setting band names and assign masks as variable to Dataset and aligning Dask chunks
        
        ds.attrs["long_name"] = band_names
        ds = ds.assign(dict(masks=scl_masks))
        
        print(ds)
        if verbose: 
            print_map(scls, "\n\n SCL ISOLATED BANDs\n")
            print(f"\n\n SCL MASKS (retrieved from interpolated DSS (-> DS))\n\n{scl_masks}\n\n")
            print_map(dss, "\n\n DSS + SET_BANDS() + AUX_BANDS_DROP()\n")
            print(f"\n\n DSS + SET_BANDS() + AUX_BANDS_DROP() + INTERPOLATION to 10M\n\n{ds}\n\n")
        
        # Masks refinement
        # The cloud and shadow masks have:
        #   0: cloud/shadow pixels
        #   1: valid pixels
        
        refined_cloud_masks = []
        refined_shadow_masks = []
        
        backgrounds = generate_seasonal_backgrounds(ds, quantile = 0.25)
        
        backgrounds.data.dask.visualize(f"{dask_graph_path}/backgrounds.svg")
        ds.data.data.dask.visualize(f"{dask_graph_path}/s2_data.svg")
        ds.masks.data.dask.visualize(f"{dask_graph_path}/s2c_masks.svg")

        for season, group in ds.groupby("season_id"):
            
            # Background computation for current season
            logger.info(f"[{season}] Processing season: {season}")
            background_path = f"{bg_output_path}/backgroundImage_{group.season_id.values[0]}.tif" 
            background = backgrounds.sel(season_id = season)
            
            if not os.path.exists(background_path):
                logger.info(f"[{season}] Start generating and saving background at {background_path}")
                            
                background.rio.to_raster(
                    background_path,
                    compress="DEFLATE",
                    num_threads="all_cpus"
                )
                
                logger.info(f"[{season}] Background saved")

            background = rioxarray.open_rasterio(background_path).squeeze()
            background = background.compute()
            
            # Masks refinement
            
            for time in group.time.values:
                
                logger.info(f"Generating and saving Sen2cor and refined cloud and shadow masks for {time}")
                c_group = group.sel(time = time)        
                    
                ref_cloud_mask_path = f"{cloud_masks_path}/{str(c_group.file_name.values)[:-4]}_cloudMediumMask.tif"
                ref_shadow_mask_path = f"{shadow_masks_path}/{str(c_group.file_name.values)[:-4]}_shadowMask.tif"

                if not os.path.exists(ref_shadow_mask_path) or not os.path.exists(ref_cloud_mask_path):
                    c_group = c_group.load()
                
                refined_cloud_mask = cloud_masks_refinement(c_group, background, cloud_coverage_threshold)
                del refined_cloud_mask.attrs['long_name']
                
                refined_shadow_mask = shadow_masks_refinement(c_group, refined_cloud_mask, cloud_coverage_threshold, image_brightness_threshold)
                del refined_shadow_mask.attrs['long_name']
            
                date = str(time)[:10]

                if not os.path.exists(ref_cloud_mask_path):

                    refined_cloud_mask.rio.to_raster(
                        ref_cloud_mask_path,
                        compress="DEFLATE",
                        num_threads="all_cpus"
                    )
                    logger.info(f"[{season}][{date}] Saved refined cloud mask at {ref_cloud_mask_path}")
                    
                else: logger.info(f"[{season}][{date}] Skipping. Already exists: {ref_cloud_mask_path}")
                
                if not os.path.exists(ref_shadow_mask_path):
                
                    refined_shadow_mask.rio.to_raster(
                        ref_shadow_mask_path,
                        compress="DEFLATE",
                        num_threads="all_cpus"
                    )
                    logger.info(f"[{season}][{date}] Saved refined shadow mask at {ref_shadow_mask_path}")
                else: logger.info(f"[{season}][{date}] Skipping. Already exists: {ref_shadow_mask_path}")
                
                refined_shadow_masks.append(refined_shadow_mask)
                refined_cloud_masks.append(refined_cloud_mask)
                
                del c_group, refined_cloud_mask, refined_shadow_mask

            del background
        
        refined_cloud_masks = xr.concat(refined_cloud_masks, dim='time')
        refined_shadow_masks = xr.concat(refined_shadow_masks, dim='time')
        
        refined_masks = xr.Dataset({
            "refined_cloud_masks" : refined_cloud_masks,
            "refined_shadow_masks" : refined_shadow_masks
        })
        
        if verbose:
            print(f"\n\n REFINED MASKS \n\n{refined_masks}\n\n")
            print(f"\n\n BACKGROUNDS \n\n{backgrounds}\n\n")
            

def masks_refinement_script() -> None:
    logged_main(
        "Cloud and Shadow Masks Refinement",
        masks_refinement,
    )