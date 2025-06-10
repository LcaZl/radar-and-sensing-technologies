import os
import dask
import logging
from scripting import load_composites, load_dems, calculate_aspect_from_dems, calculate_slope_from_dems
import pandas as pd
import xarray as xr
import numpy as np
from scripting import logged_main, monitor_memory
import dask 
import dask.distributed
from scripting import load_features
from svm_pipeline import load_SVMS
import time
import dask.array as da
from ._generate_lc_map_utils import (
        expand_bands_and_reduce, 
        create_empty_tif, 
        load_map_and_plot, 
        load_map_with_probs_and_plot,
        process_chunk
    )

from svm_pipeline import get_scaler
from threading import Thread
from sklearn.decomposition import PCA

def generate_lc_map(**parameters) -> None:

    performance_file = f"{parameters['log_dir']}/{parameters['log_name'][:-4]}.csv"
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    with dask.distributed.Client(
        processes=False,
        threads_per_worker= (os.cpu_count() or 4),
    ) as client:
        
        logger = logging.getLogger("svm-cl-pipeline")

        logger.info(f"Dask dashboard: {client.dashboard_link}")
        logger.info(f"Generating classification map for {parameters['composites_year']}")
        
        os.makedirs(parameters["output_path"], exist_ok=True)
        parameters["classified_image_path"] = f"{parameters['output_path']}/{parameters['tile_id']}_{parameters['composites_year']}_{parameters['chunks_limit']}r"
        png_images_path = f"{parameters['output_path']}/images"
        os.makedirs(png_images_path, exist_ok=True)

        # Load data
        
        composites = load_composites(parameters["composites_path"], year=parameters["composites_year"], tile=parameters["tile_id"])
        dems = load_dems(parameters["dems_path"] ,year=parameters["dems_year"], tile=parameters["tile_id"])
        slope = calculate_slope_from_dems(dems.band_data)
        aspect = calculate_aspect_from_dems(dems.band_data)
        glcm_features = load_features(parameters['features_path'])
        id2label = {id : row['description'] for id, row in pd.read_csv(parameters["labels_path"]).iterrows()}
        label2id = {label : id for id, label in id2label.items()}
        
        dataset = composites.assign({
                            "dems":dems.band_data,
                            "slopes":slope,
                            "aspects":aspect}).sel(tile=parameters["tile_id"])
        
        dataset = dataset.assign({
            f_name : feature.isel(time=0) for f_name, feature in glcm_features.items() # Only January GLCM features are kept. 
        })
        
        print(f"Dems:\n{dems}\n\n")
        print(f"Aspect:\n{aspect}\n\n")
        print(f"Slope:\n{slope}\n\n")
        print(f"Composites:\n{composites}\n\n")
        print(f"Dataset:\n{dataset}\n\n")

        # Select features to keep and flatten the bands
        svm = load_SVMS(parameters["model_path"])
        features = list(svm.preprocessing_metadata.keys())
        dataset_flattened = expand_bands_and_reduce(dataset, features)
        to_exlude = set(['ground_truth_index', 'ground_truth_label', 'split', 'x', 'y'])
        dataset_selected = dataset_flattened[svm.features_selected]
        
        # Chunk accordingly to parameters
        chunk_size = parameters["chunk_size"]
        dataset_selected = dataset_selected.chunk({"x": chunk_size, "y": chunk_size})

        # Load preprocessing structures from SVMS class
        args = {}
        
        scalers = {f : get_scaler(info["method"], info["params"]) for f, info in svm.preprocessing_metadata.items()}
        pca_model = None
        if svm.pca is not None:
            pca_model = PCA()
            pca_model.components_ = svm.pca["params"]["components_"]
            pca_model.n_components_ = svm.pca["params"]["n_components_"]
            pca_model.explained_variance_ = svm.pca["params"]["explained_variance_"]
            pca_model.singular_values_ = svm.pca["params"]["singular_values_"]
            pca_model.mean_ = svm.pca["params"]["mean_"]
            pca_model.n_samples_ = svm.pca["params"]["n_samples_"]
            pca_model.noise_variance_ = svm.pca["params"]["noise_variance_"]
            pca_model.n_features_in_ = svm.pca["params"]["n_features_in_"]
            pca_model.feature_names_in_ = svm.pca["params"]["feature_names_in_"]

        
        # Create output images accordingly to parameters
        
        feature_names = list(dataset_selected.data_vars.keys())
        height, width = dataset_selected.x.shape[0], dataset.y.shape[0]
        transform = dataset_selected.rio.transform()
        crs = dataset_selected.rio.crs
        
        # Specify the models to use and thus the number of output images generated
        svms_to_use = parameters["svms_to_use"]
        
        prefix = f"{parameters['tile_id']}_{parameters['composites_year']}_{parameters['chunks_limit']}r"
        output_paths = {}
        if 'svmMc' in svms_to_use:
            output_paths["svmMc"] = f"{parameters['output_path']}/{prefix}_predictions_SvmMc.tif"
        if 'svmMcCal' in svms_to_use:
            output_paths["svmMcCal"] = f"{parameters['output_path']}/{prefix}_predictions_SvmMcCal.tif"
        if 'svmsBin' in svms_to_use:
            output_paths["svmsBin"] = f"{parameters['output_path']}/{prefix}_predictions_SvmsBin.tif"
        if 'svmsBinSoftmax' in svms_to_use:
            output_paths["svmsBinSoftmax"] = f"{parameters['output_path']}/{prefix}_predictions_SvmsBinSoftmax.tif"
        
        band_labels = svm.classes.tolist() + ["labels"] # Custom bands names for the proability map (+ label)
        logger.info(f"Band labels ({len(band_labels)}): {band_labels}")
        
        for name, path in output_paths.items():
            create_empty_tif(path, width, height, len(band_labels), "uint8", transform, crs, band_labels)

        # Generate labels mapper
        
        labels_df =  pd.read_csv(parameters["labels_path"])
        id2label = {row['LC_code'] : row['description'] for id, row in labels_df.iterrows()}
        label2id = {label : id for id, label in id2label.items()}
        label_mapper = np.vectorize(lambda label: label2id[label])
        
        args["label_mapper"] = label_mapper
        args["pca"] = pca_model
        args["svm"] = svm
        
        # Generate tasks
        limited = parameters["chunks_limit"] is not None
        limit = parameters["chunks_limit"]
        
        logger.info("Start processing dataset chunks:")
        logger.info(f" - Limit: {limit}")
        i = 0
        
        for x_start in range(0, width, chunk_size):
            for y_start in range(0, height, chunk_size):
                
                start = time.perf_counter()
                logger.info(f"Processing chunk {i}")
                
                patch = dataset_selected.isel(
                    x=slice(x_start, x_start + chunk_size),
                    y=slice(y_start, y_start + chunk_size)
                )
                
                features_array = da.stack([patch[var] for var in feature_names]).compute()

                args['task_id'] = i
                process_chunk(
                    features_array, feature_names, scalers, args ,x_start, y_start, output_paths, svms_to_use
                )
                
                elapsed = time.perf_counter() - start
                logger.info(f"Chunk {i} processed in {elapsed:.2f} seconds")
                
                del features_array
                i += 1
                
                if limited and limit == i:
                    break
                
            if limited and limit == i:
                break
                    

        for name, path in output_paths.items():
            _ = load_map_with_probs_and_plot(path, band_labels, labels_df, name, png_images_path)  
            
        if parameters['baseline_lc_map_path'] is not None:
            _ = load_map_and_plot(parameters['baseline_lc_map_path'], labels_df, "Previous Pipeline predictions", png_images_path)
            
def generate_lc_map_script() -> None:
    
    logged_main(
        "Use SVM for classification",
        generate_lc_map
    )