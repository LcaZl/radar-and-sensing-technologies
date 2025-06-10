
import dask.distributed
import os
import logging
from pyproj import CRS
import pandas as pd
from threading import Thread
import numpy as np
import cv2

from scripting import (
    load_composites, 
    load_dems,
    calculate_aspect_from_dems,
    calculate_slope_from_dems,
    load_points,
    load_lc_points,
    load_features,
    logged_main,
    monitor_memory,
    create_geodf_for_lc_map
)

from ._generate_dataset_utils import (
    enhance_dataset,
    extract_features_for_points,
    apply_erosion_and_report,
    convert_sav_to_csv
)

from ._dataset_inspection import (
    generate_maps,
    visual_match_verification_l1,
    visual_match_verification_l2,
    plot_distance_histogram,
    inspect_class_distribution,
    verify_coordinates
)

def generate_dataset(
    **parameters
) -> None:
    
    performance_file = f"{parameters['log_dir']}/{parameters['log_name'][:-4]}.csv"
    monitoring_thread = Thread(target=monitor_memory, args=(performance_file, 1), daemon=True)
    monitoring_thread.start()
    
    with dask.distributed.Client(
        processes=False,
        threads_per_worker=(os.cpu_count() or 2),
    ) as client:
        
        logger = logging.getLogger("generate-dataset")
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        
        if parameters["convert_sav_dataset"] == True:
            
            output_path = f"{parameters['output_path']}/{os.path.basename(parameters['sav_dataset_path']).replace('.sav', '.csv')}"
            
            dt = convert_sav_to_csv(
                sav_path=parameters["sav_dataset_path"],
                parameters=parameters,
                csv_path=output_path
            )
            logger.info(f"Converted .sav dataset saved to {output_path}")
            return;
    
        points_dataset_name = os.path.basename(parameters['points_dataset_path']).split("_")[-1][:-4]
        if parameters['dataset_type'] == 'enhanced':
            
            dataset_id = f"T{parameters['tile_id']}_f{parameters['features_date']}_{points_dataset_name}_{parameters['enhanced_type']}_{parameters['dataset_type']}"
            
            if parameters["apply_erosion"] == True:
                dataset_id = f"{dataset_id}_eroded"
                
        elif parameters['dataset_type'] == 'fullLC':
            
            dataset_id = f"T{parameters['tile_id']}_f{parameters['features_date']}_{points_dataset_name}_{parameters['samples_per_class']}_{parameters['dataset_type']}"
            
            if parameters["apply_erosion"] == True:
                dataset_id = f"{dataset_id}_eroded"
                
        elif parameters['dataset_type'] == 'std':
            dataset_id = f"T{parameters['tile_id']}_f{parameters['features_date']}_{points_dataset_name}_{parameters['dataset_type']}"
        else:
            raise ValueError("Provided dataset type not supported. Choose one of: std, fullLC or enhanced.")
        
        dataset_path = f"{parameters['output_path']}/{dataset_id}.csv"
        features_path = f"{parameters['features_path']}/{parameters['features_date']}/*.tif"

        os.makedirs(parameters['report_path'], exist_ok=True)
        curr_reports_path = f"{parameters['report_path']}/{dataset_id}_reports"
        
        os.makedirs(curr_reports_path, exist_ok=True)
        lc_points_map_path = f"{curr_reports_path}/lcmap_points.html"
        points_map_path = f"{curr_reports_path}/shapefile_points.html"
        erosion_report_path = f"{curr_reports_path}/erosion_report.png"
        
        os.makedirs(parameters['output_path'], exist_ok=True)

        # Load pre computed features
        composites = load_composites(parameters["composites_path"], year=parameters["composites_year"], tile=parameters["tile_id"])
        dems = load_dems(parameters["dems_path"],year=parameters["dems_year"], tile=parameters["tile_id"])
        slope = calculate_slope_from_dems(dems.band_data)
        aspect = calculate_aspect_from_dems(dems.band_data)
        features = load_features(features_path)
        
        features_dataset = composites.assign({
                            "dems":dems.band_data,
                            "slopes":slope,
                            "aspects":aspect}).sel(tile=parameters["tile_id"])
        
        features_dataset = features_dataset.assign({
            f_name : feature.isel(time=0) for f_name, feature in features.items() # Only January GLCM features are kept. 
        })
        
        print(f"Dems:\n{dems}\n\n")
        print(f"Aspect:\n{aspect}\n\n")
        print(f"Slope:\n{slope}\n\n")
        print(f"Composites:\n{composites}\n\n")
        print(f"Features:\n{features_dataset}\n\n")
        
        composites_crs = CRS.from_wkt(features_dataset.spatial_ref.attrs["crs_wkt"]).to_epsg()
    
        # Load points from shape file
        logger.info("Loading UniTN/Unitn+Polimi points ...")
        points_df, labels_df, tile_geometry = load_points(parameters, features_dataset, composites_crs)
        lccode2label = {row['LC_code'] : row['description'] for id, row in labels_df.iterrows()}
        id2lccode = {row['internal_code'] : row['LC_code'] for id, row in labels_df.iterrows()}
        points_df['class_id'] = points_df['class_id'].map(id2lccode)

        # Load the requested amount of point from Land Cover map

        logger.info("Selecting dataset points ...")
        if parameters["dataset_type"] != 'std':
            
            logger.info("Loading land cover map points ...")
    
            lc_map_xr = load_lc_points(parameters)

            if parameters["apply_erosion"] == True:
                
                logger.info("Applying erosion ...")
                config = {
                    'kernel': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), #np.ones((3,3), np.uint8),         # Size of the erosion kernel
                    'iterations': 1             # Number of iterations for erosion
                }

                classes_masks = apply_erosion_and_report(lc_map_xr, lccode2label, config, output_path = erosion_report_path)
            else:
                classes_masks = {id : (lc_map_xr == id).astype(np.uint8) for id in lccode2label.keys() if (lc_map_xr == id).mean() > 0}
                
            class_counts = points_df['class_id'].value_counts()
            max_class_samples = class_counts.max()
            avg_class_samples = int(class_counts.mean())
            
            if parameters["dataset_type"] != 'fullLC':
                # Determine target samples per class
                if parameters["enhanced_type"] == "max":
                    parameters["samples_per_class"] = max_class_samples
                elif parameters["enhanced_type"] == "avg":
                    parameters["samples_per_class"] = avg_class_samples
                else:
                    raise ValueError("Invalid value for enhanced_type. Choose 'max' or 'avg'.")

            points_lc_df = create_geodf_for_lc_map(classes_masks, parameters, points_df, features_dataset, lccode2label, composites_crs)
        
        if parameters["dataset_type"] == "enhanced":
            
            ids_shp = set(points_df['class_id'].unique())
            ids_lcmap = set(points_lc_df['class_id'].unique())
            print(f"Classes in shape file: {ids_shp}")
            print(f"Classes in LC map: {ids_lcmap}")
            common_ids = ids_shp & ids_lcmap
            print(f"Common ids: {common_ids}")
            points_df = points_df[points_df['class_id'].isin(common_ids)]
            points_lc_df = points_lc_df[points_lc_df['class_id'].isin(common_ids)]
            print(f"Final classes available: {points_lc_df['class'].unique()}")
            dataset = pd.concat([points_df, points_lc_df[["x","y","class_id","class","split"]]])
            
            dataset = enhance_dataset(points_df, points_lc_df, target_class_col="class_id", samples_per_class=parameters["samples_per_class"])
            inspect_class_distribution(dataset, 
                                    class_column='class', 
                                    output_path = f"{curr_reports_path}/classes_distribution.png",
                                    title = f"Enhanced ({parameters['enhanced_type']}) dataset class distribution - Total points: {len(dataset)}")

        elif parameters["dataset_type"] == "fullLC":
            
            ids_shp = set(points_df['class_id'].unique())
            ids_lcmap = set(points_lc_df['class_id'].unique())
            print(f"Classes in shape file: {ids_shp}")
            print(f"Classes in LC map: {ids_lcmap}")
            common_ids = ids_shp & ids_lcmap
            print(f"Common ids: {common_ids}")
            points_df = points_df[points_df['class_id'].isin(common_ids)]
            points_lc_df = points_lc_df[points_lc_df['class_id'].isin(common_ids)]
            print(f"Final classes available: {points_lc_df['class'].unique()}")
            dataset = pd.concat([points_df, points_lc_df[["x","y","class_id","class","split"]]])
            
            inspect_class_distribution(dataset[dataset['split'] == 'train'], 
                                    class_column='class', 
                                    output_path = f"{curr_reports_path}/train_classes_distribution.png",
                                    title = f"FullLC dataset train split class distribution - Total points: {len(dataset[dataset['split'] == 'train'])}")
            
            inspect_class_distribution(dataset[dataset['split'] == 'test'], 
                                    class_column='class', 
                                    output_path = f"{curr_reports_path}/test_classes_distribution.png",
                                    title = f"FullLC dataset test/val split class distribution - Total points: {len(dataset[dataset['split'] == 'test'])}")

        elif parameters["dataset_type"] == "std":
            
            dataset = points_df[["x","y","class_id","class","split"]]
            inspect_class_distribution(dataset, 
                                    class_column='class', 
                                    output_path = f"{curr_reports_path}/classes_distribution.png",
                                    title = f"Standard dataset class distribution - Total points: {len(dataset)}")

        logger.info("Generating final dataset ...")
        
        extract_features_for_points(features_dataset, dataset, dataset_path)
        
        logger.info(f"Saving dataset maps ...")
            
        if parameters["dataset_type"] == "fullLC" or parameters["dataset_type"] == "enhanced":

                _ = generate_maps(points=points_lc_df, 
                                        tile_geometry = tile_geometry,
                                                output_file=lc_points_map_path)

        _ = generate_maps(points=points_df, 
                                tile_geometry = tile_geometry,
                                        output_file=points_map_path)
        verified_points = verify_coordinates(points_df.copy(), features_dataset)

        print(verified_points[["geometry", "x", "y", "recalc_lon", "recalc_lat", "diff_lon", "diff_lat"]].head())
        print(f"Average longitude difference: {verified_points['diff_lon'].mean()}")
        print(f"Min/Max longitude difference: {verified_points['diff_lon'].min()} - {verified_points['diff_lon'].max()}")
        print(f"Average latitude difference: {verified_points['diff_lat'].mean()}")
        print(f"Min/Max latitude difference: {verified_points['diff_lat'].min()} - {verified_points['diff_lat'].max()}")

        plot_distance_histogram(verified_points, save_path=curr_reports_path)
        visual_match_verification_l1(verified_points, save_path=curr_reports_path)
        visual_match_verification_l2(verified_points, save_path=curr_reports_path)

        if parameters["dataset_type"] == "fullLC" or parameters["dataset_type"] == "enhanced":

                _ = generate_maps(points=points_lc_df, 
                                        tile_geometry = tile_geometry,
                                                output_file=lc_points_map_path)

        _ = generate_maps(points=points_df, 
                                tile_geometry = tile_geometry,
                                        output_file=points_map_path)
                
def generate_dataset_script() -> None:
    logged_main(
        "Generating dataset",
        generate_dataset,
    )