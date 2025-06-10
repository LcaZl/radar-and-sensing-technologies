import os
import logging
import dask.distributed
import xarray as xr
import numpy as np
from threading import Thread
import rioxarray

from ._cloud_restoration_utils import (
    shadow_adjustment,
    restore_cloud_shadow_xr
)

from scripting import (
    load_composites,
    load_dems,
    calculate_aspect_from_dems,
    calculate_slope_from_dems,
    logged_main,
    monitor_memory,

)
    
def cloud_restoration(
    **parameters
) -> None:
    
    performance_file = f"{parameters['log_dir']}/{parameters['log_name'][:-4]}.csv"
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    with dask.distributed.Client(
        processes=False,
        threads_per_worker=(os.cpu_count() or 2),
    ) as client:
        
        # Parameters setup
        
        logger = logging.getLogger("cloud-restoration")
        tile_id = parameters["tile_id"]
        
        logger.info(f"Cloud Restoration for tile {tile_id}")
        logger.info(f"Dask dashboard: {client.dashboard_link}")

        composites_restored_path = f"{parameters['output_path']}/{parameters['composites_path'].split('/')[-2]}_restored"
        composites_restored_corr_path = f"{parameters['output_path']}/{parameters['composites_path'].split('/')[-2]}_restored_adj"
        dask_graph_path = f"{parameters['output_path']}/dask_graphs/cloud_restoration.svg"
        
        os.makedirs(f"{parameters['output_path']}/dask_graphs", exist_ok=True)
        os.makedirs(composites_restored_path, exist_ok=True)
        os.makedirs(composites_restored_corr_path, exist_ok=True)
        
        # Loading composites dataset and DEMs
        
        logger.info("Loading composites.")
        composites = load_composites(parameters['composites_path'], parameters["composites_year"], tile_id)
        if parameters["verbose"]:
            print(f"Composites raw:\n{composites}\n\n")

        dems = load_dems(parameters['dems_path'], parameters["dems_year"], tile_id)
        
        # Compute slopes and aspects
        
        slope = calculate_slope_from_dems(dems.band_data)
        aspect = calculate_aspect_from_dems(dems.band_data)
        
        composites = composites.assign({
            "dems":dems.band_data,
            "slopes":slope,
            "aspects":aspect})
        
        if parameters["verbose"]:
            print(f"Dems:\n{dems}\n\n")
            print(f"Aspect:\n{aspect}\n\n")
            print(f"Slope:\n{slope}\n\n")
            print(f"Composites Dataset:\n{composites}\n\n")
        
        # Restore cloud/shadow pixels
        
        logger.info(f"Restoring cloud/shadow pixels ...")
        
        composites["band_data_restored"] = composites.band_data.groupby("band").map(restore_cloud_shadow_xr)

        if parameters["verbose"]:
            logger.info(f"Composites Restored:\n{composites.band_data_restored}\n\n")
        
        if parameters["verbose"]:
            logger.info(f"Composites with band data restored:\n{composites}\n\n")

        # Save the restored and the restored + adjusted composites
        
        cr_paths = []
        crs_paths = []
        for time_idx in range(composites.sizes['time']):
            
            name = composites.file_name.isel(time=time_idx).values
            name = str(name).split('_')
            name_restored = f"{name[0]}Restored_{name[1]}_{name[2]}"
            name_adj = f"{name[0]}RestoredAdj_{name[1]}_{name[2]}"
            
            cr_paths.append(
                os.path.join(composites_restored_path,name_restored)
            )
            
            crs_paths.append(
                os.path.join(composites_restored_corr_path,name_adj)
            )
            
        slopes = composites.slopes.sel(tile=tile_id).values

        for time_idx, comp_rest_path, comp_rest_corr_path in zip(range(composites.sizes['time']), cr_paths, crs_paths) :
            
            if not os.path.exists(comp_rest_path):
                
                composite_rest = composites.band_data_restored.isel(time=time_idx)
            
                logger.info(f"Saving Restored Image at {comp_rest_path} ...")
                if parameters["verbose"]:
                    logger.info(f"Composite:\n{composite_rest}\n")
                            
                composite_rest.rio.to_raster(comp_rest_path, compress="DEFLATE", num_threads="all_cpus")
                logger.info(f"Saved Restored Image.")    

            composite_rest = rioxarray.open_rasterio(comp_rest_path).squeeze()
            
            if not os.path.exists(comp_rest_corr_path):
                   
                composite_rest_val = composite_rest.values
                
                logger.info(f"Correcting restored image ...")
                
                composite_rest_corr = shadow_adjustment(composite_rest_val, slopes)
                
                composite_rest_corr_da = xr.DataArray(
                    composite_rest_corr.squeeze(),
                    dims=composite_rest.dims,
                    coords=composite_rest.coords,
                )
            
                logger.info(f"Saving Restored Corrected Image at {comp_rest_corr_path} ...")
                
                if parameters["verbose"]:
                    print(f"Composite restored corrected:\n{composite_rest_corr_da}\n")
                    
                composite_rest_corr_da.rio.to_raster(comp_rest_corr_path, compress="DEFLATE", num_threads="all_cpus")
                logger.info(f"Saved Restored Corrected Image.")
        
            else:
                logger.info(f"Corrected Restored Composite already exists, skipping: {comp_rest_corr_path}")
        
            
        # Save dask graph
        #logger.info(f"Saved Dask graph at : {dask_graph_path}")
        #composites.band_data_restored.data.dask.visualize(dask_graph_path)



def cloud_restoration_script() -> None:
    logged_main(
        "Cloud Restoration",
        cloud_restoration
    )