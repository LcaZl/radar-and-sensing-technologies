        print("\n\nInitial xarray dataset flattened:\n\n",dataset_flattened)
        print("\n\nDataset flattened selected:\n\n",dataset_selected)

        args = {}
        args['label_mapper'] = label_mapper
        
        if svm.pca_metadata is not None:
            pca_model = PCA()
            pca_model.components_ = svm.pca_metadata["params"]["components_"]
            pca_model.n_components_ = svm.pca_metadata["params"]["n_components_"]
            pca_model.explained_variance_ = svm.pca_metadata["params"]["explained_variance_"]
            pca_model.singular_values_ = svm.pca_metadata["params"]["singular_values_"]
            pca_model.mean_ = svm.pca_metadata["params"]["mean_"]
            pca_model.n_samples_ = svm.pca_metadata["params"]["n_samples_"]
            pca_model.noise_variance_ = svm.pca_metadata["params"]["noise_variance_"]
            pca_model.n_features_in_ = svm.pca_metadata["params"]["n_features_in_"]
            pca_model.feature_names_in_ = svm.pca_metadata["params"]["feature_names_in_"]

            args["pca"] = pca_model

        # Load class mappings

        # Define the patch size
        chunk_size = parameters["chunk_size"]  # Each chunk is 1098x1098 -> 100 chunks
        dataset_selected = dataset_selected.chunk({"x": chunk_size, "y": chunk_size})
        feature_names = list(dataset_selected.data_vars.keys())
        labels_df =  pd.read_csv(parameters["labels_path"])
        id2label = {row['LC_code'] : row['description'] for id, row in labels_df.iterrows()}
        label2id = {label : id for id, label in id2label.items()}
        label_mapper = np.vectorize(lambda label: label2id[label])

        args["label_mapper"] = label_mapper
        tasks = []
        i = 0
        limited = parameters["chunks_limit"] is not None
        limit = parameters["chunks_limit"]
        chunk_positions = []  # Store the chunk positions in the same order as tasks

        logger.info("Processing dataset chunks:")
        logger.info(f" - Limit: {limit}")
        
        for x_start in range(0, dataset_selected.sizes["x"], chunk_size):
            for y_start in range(0, dataset_selected.sizes["y"], chunk_size):

                if limited and i == limit:
                    break
                
                logger.info(f"Chunk {i+1} at x={x_start}-{x_start+chunk_size}, y={y_start}-{y_start+chunk_size} added to delayed tasks.")

                patch = dataset_selected.isel(
                    x=slice(x_start, x_start + chunk_size),
                    y=slice(y_start, y_start + chunk_size)
                )

                features_array = da.stack([patch[var] for var in feature_names])
                x_indexes = np.arange(x_start, min(x_start + chunk_size, 10980))
                y_indexes = np.arange(y_start, min(y_start + chunk_size, 10980))

                task = delayed(process_chunk)(
                    features_array, feature_names, scalers, svm, args, x_indexes, y_indexes
                )
                tasks.append(task)
                chunk_positions.append((x_start, y_start))

                i += 1
            if limited and i == limit:
                break 
                
        # Execute computations
        logger.info(f"Predictions - Computing {len(tasks)} Dask's task")
        results = dask.compute(*tasks)
        
        logger.info("Tasks computed")
        
        band_dim_labels = svm.classes.tolist() + ["labels"]

        def create_output_image():
            output_shape = (len(band_dim_labels), 10980, 10980)  # Define the full image size
            return xr.DataArray(
                np.full(output_shape, np.nan, dtype=np.uint8),  # Use uint8 for classification
                dims=("band", "y", "x"),
                coords={"band": band_dim_labels, "y": dataset["y"], "x": dataset["x"]},
            ).assign_coords({"spatial_ref": dataset.spatial_ref})
            
        output_images = {
            "predictions_SvmMc": create_output_image(),
            "predictions_SvmMcCal": create_output_image(),
            "predictions_SvmsBin": create_output_image(),
            "predictions_SvmsBinSm": create_output_image(),
        }

        print("Saving land cover maps to disk")

        # Ensure the number of results matches the chunk positions
        if len(results) != len(chunk_positions):
            print("Mismatch between computed results and expected chunk positions!")
            raise ValueError("Computed results do not match expected chunk positions.")

        # Assign values based on chunk positions
        for i, ((predictions_SvmMc, 
                predictions_SvmMcCal, 
                predictions_SvmsBin, 
                #predictions_SvmsBinSm,
                predProb_SvmMc, 
                predProb_SvmMcCal,
                predProb_SvmsBin,
                #predProb_SvmsBinSm
                ), 
                (x_start, y_start)) in enumerate(zip(results, chunk_positions)):
            
            print(f"Writing results for chunk {i+1} at x={x_start}-{x_start+chunk_size}, y={y_start}-{y_start+chunk_size}")

            # Get Numpy arrays from Dask arrays
            pred_svm_mc = predictions_SvmMc.compute()
            pred_svm_mc_cal = predictions_SvmMcCal.compute()
            pred_svms_bin = predictions_SvmsBin.compute()
            #pred_svms_bin_sm = predictions_SvmsBinSm.compute()
            predProb_svm_mc = predProb_SvmMc.compute()
            predProb_svm_mc_cal = predProb_SvmMcCal.compute()
            predProb_svms_bin = predProb_SvmsBin.compute()
            #predProb_svms_bin_sm = predProb_SvmsBinSm.compute()

            output_images["predictions_SvmMc"].sel(band = 'labels').isel(
                y=slice(y_start, y_start + chunk_size), x=slice(x_start, x_start + chunk_size)
            ).data[:] = label_mapper(pred_svm_mc)

            output_images["predictions_SvmMcCal"].sel(band = 'labels').isel(
                y=slice(y_start, y_start + chunk_size), x=slice(x_start, x_start + chunk_size)
            ).data[:] = label_mapper(pred_svm_mc_cal)

            output_images["predictions_SvmsBin"].sel(band = 'labels').isel(
                y=slice(y_start, y_start + chunk_size), x=slice(x_start, x_start + chunk_size)
            ).data[:] = label_mapper(pred_svms_bin)

            #output_images["predictions_SvmsBinSm"].sel(band = 'labels').isel(
                #y=slice(y_start, y_start + chunk_size), x=slice(x_start, x_start + chunk_size)
            #).data[:] = label_mapper(pred_svms_bin_sm)

            # Write class-wise probabilities
            for class_index, class_name in enumerate(svm.classes):
                output_images["predictions_SvmMc"].sel(band=class_name).isel(
                    y=slice(y_start, y_start + chunk_size),
                    x=slice(x_start, x_start + chunk_size)
                ).data[:] = (predProb_svm_mc[:, :, class_index] * 255).astype(np.uint8)

                output_images["predictions_SvmMcCal"].sel(band=class_name).isel(
                    y=slice(y_start, y_start + chunk_size),
                    x=slice(x_start, x_start + chunk_size)
                ).data[:] = (predProb_svm_mc_cal[:, :, class_index] * 255).astype(np.uint8)

                output_images["predictions_SvmsBin"].sel(band=class_name).isel(
                    y=slice(y_start, y_start + chunk_size),
                    x=slice(x_start, x_start + chunk_size)
                ).data[:] = (predProb_svms_bin[:, :, class_index] * 255).astype(np.uint8)
                
                #output_images["predictions_SvmsBinSm"].sel(band=class_name).isel(
                    #y=slice(y_start, y_start + chunk_size),
                    #x=slice(x_start, x_start + chunk_size)
                #).data[:] = (predProb_svms_bin_sm[:, :, class_index] * 255).astype(np.uint8)
                
        print("Writing complete. Now saving images.")

        storage_paths = []
        for name, image in output_images.items():
            
            output_path = f"{parameters['classified_image_path']}_{name}.tif"
            storage_paths.append(output_path)
            
            image.rio.to_raster(output_path)            
            print(f"Saved classified image: {output_path}")

        print("All classification GeoTIFF images saved successfully.")

        for path, name in zip(storage_paths, output_images.keys()):
            _ = load_map_with_probs_and_plot(path, band_dim_labels, name, png_images_path)    

        if parameters['baseline_lc_map_path'] is not None:
            _ = load_map_and_plot()