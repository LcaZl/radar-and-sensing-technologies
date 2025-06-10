
import logging
import dask
import dask.distributed
import os
import xarray as xr
import numpy as np
import rioxarray
from threading import Thread

from scripting import (
    load_composites,
    load_dems,
    calculate_aspect_from_dems,
    calculate_slope_from_dems,
    logged_main,
    print_dict,
    monitor_memory
)

from ._generate_features_utils import computeGLCM

from generate_features import (
    get_NDVI,
    get_SAVI,
    get_NDMI,
    get_NDWI,
    get_NDBI,
    get_NDSI,
    get_BSI,
    get_sobel,
    computeGLCM,
    get_NDLI,
    get_NDVI705,
    get_GNDVI,
    get_EVI2,
    get_GLI,
    get_NDYI
)

def generate_features(    
    **parameters
) -> None:
    
    performance_file = f"{parameters['log_dir']}/{parameters['log_name'][:-4]}.csv"
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    with dask.distributed.Client(
        processes=False,
        threads_per_worker=(os.cpu_count() or 2),
    ) as client:
        print_dict(parameters)
        logger = logging.getLogger("process-features")
        logger.info(f"Dask dashboard: {client.dashboard_link}")

        # Parameters
        features_date = parameters["features_date"]
        
        features_year, features_month = features_date.split("-")
        if len(features_month) > 1 and features_month[0] == '0':
            features_month = features_month[1]

        features_folder_path = f"{parameters['output_path']}/{str(features_date)}"         

        os.makedirs(parameters['output_path'], exist_ok=True)
        os.makedirs(features_folder_path, exist_ok=True)

        # Load composites, dems, slope and aspects
        
        logger.info("Loading composites.")
        composites = load_composites(parameters['composites_path'], year=features_year, tile=parameters['tile_id'], month = features_month)
        
        dems = load_dems(parameters['dems_path'],year=parameters['dems_year'], tile=parameters['tile_id']).band_data
        slope = calculate_slope_from_dems(dems)
        aspect = calculate_aspect_from_dems(dems)
        
        raw_features = composites.assign({
            "dems":dems,
            "slopes":slope,
            "aspects":aspect}) # Shape [tile, time, band, x, y]
        
        if parameters['verbose']:
            print(f"Dems:\n{dems}\n\n")
            print(f"Aspect:\n{aspect}\n\n")
            print(f"Slope:\n{slope}\n\n")
            
        # Selecting the band specified in parameters for GLCM computation
        logger.info(f"Computing data ...")
        raw_features = raw_features.sel(band = parameters['band_to_use'], tile = parameters['tile_id']).band_data.squeeze() # Shape [time, 10980, 10980]
        
        if parameters['verbose']:
            print(f"Composites band data:\n{raw_features}\n")
            
        # GLCM & related features
        if parameters["compute_GLCM"] == True:
            logger.info(f"Generating GLCM and related features from time: {features_date} ...")
                    
            glcm_features_dict = computeGLCM(raw_features.values, parameters['window_size'], parameters['glcm_distance'], parameters['glcm_angle'], parameters['batch_size'])

            glcm_features = {
                feature_name: xr.DataArray(
                    glcm_features_dict[feature_name][np.newaxis, :, :],  # Add a new axis for the optional time dimension
                    dims=["time", "y", "x"],
                    coords={
                        "y": raw_features.coords['y'],
                        "x": raw_features.coords['x'],
                        "time": raw_features.coords['time'],
                        "spatial_ref": raw_features.spatial_ref,
                    },
                    name=feature_name,
                ).astype(np.float32)
                for feature_name in list(glcm_features_dict.keys())
            }
                
            if parameters['verbose']:
                print(f"\nGLCM features:\n{glcm_features}\n\n")

            # Save GLCM features
            
            logger.info(f"Saving features ...")      

            for f_name, f_dt in glcm_features.items(): 
                    
                f_path = f"{features_folder_path}/GLCM{f_name}_{str(features_date)[:7]}.tif"
                logger.info(f"Saving {f_name} feature to {f_path}")

                if not os.path.exists(f_path):    
                    f_dt.rio.to_raster(f_path, compress="DEFLATE", num_threads="all_cpus")
                    logger.info(f"Feature {f_name} saved.")
                else:
                    logger.info(f"Features {f_name} at {features_date} already exists. Remove it to recompute.")
            
        # Generate other features

        features_funcs = {
            "NDVI" : get_NDVI,
            "NDVI705" : get_NDVI705,
            "GNDVI" : get_GNDVI,
            "NDYI" : get_NDYI,
            "EVI" : get_EVI2,
            "SAVI" : get_SAVI,
            "NDMI" : get_NDMI,
            "NDWI" : get_NDWI,
            "NDBI" : get_NDBI,
            "NDSI" : get_NDSI,
            "BSI" : get_BSI,
            "NDLI" : get_NDLI,
            "Sobel" : get_sobel,
            "GLI" : get_GLI
        }
    
        all_features = {}
        #all_features_values = {}
        composites.band_data.compute()
        
        for feature_name, feature_func in features_funcs.items():
            print(f"\nProcessing feature {feature_name}")
            
            feature_path = f"{features_folder_path}/{feature_name}_{str(features_date)[:7]}.tif"
            print(f"Path: {feature_path}")

            if not os.path.exists(feature_path):
                print("Computing ...")
                
                if feature_name == "Sobel":
                    feature = feature_func(all_features['NDVI'])
                else:
                    feature = feature_func(composites.band_data)
                feature.rio.to_raster(feature_path, compress="DEFLATE", num_threads="all_cpus")
            #else:
                #print("Loading from disk ...")
                #feature = rioxarray.open_rasterio(feature_path)
                
            all_features[feature_name] = feature
            #all_features_values[feature_name] = feature.values
            
            
def generate_features_script() -> None:
    logged_main(
        "Generating features",
        generate_features
    )