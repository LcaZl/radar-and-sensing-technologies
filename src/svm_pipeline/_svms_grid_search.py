import itertools
import yaml
import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import warnings
import itertools
import os

from ._pipeline_utils import save_experiment_result, standardize_dict, reload_and_apply_PCA
from ._svms_evaluation import test_svm, train_binary_svms, train_svm
from ._features_preprocessing import preprocess_features, split_dataset, get_scaler
from ._features_selection import features_selection, apply_pca

from scripting import (
    print_dict, print_list, print_dataframe
)

def filter_grid_search_combinations(grid_search_params: dict) -> list[dict]:
    """
    Filters out redundant or invalid parameter combinations from a grid search.

    This function evaluates all combinations of grid search parameters and applies
    three levels of pruning:
    1. Removes configurations that would result in empty feature sets during feature selection.
    2. Removes configurations where PCA would be applied to no valid features.
    3. Collapses redundant combinations by setting irrelevant parameters to None and
       comparing canonical forms of parameter sets.

    Parameters
    ----------
    grid_search_params : dict
        Dictionary of parameters prefixed with 'gs_' and their list of values.

    Returns
    -------
    list of dict
        List of valid, distinct parameter combinations to be tested.
    """

    param_keys = list(grid_search_params.keys())
    all_combos = list(itertools.product(*grid_search_params.values()))

    seen = set()
    filtered_combos = []

    for combo in all_combos:
        # Build a dict for easier reading
        combo_dict = dict(zip(param_keys, combo))
        dataset_path = combo_dict.get('gs_dataset_path', '')
        dataset_basename = os.path.basename(dataset_path).lower()
        is_dataset_sav = (dataset_basename == "dataset.csv")

        # -- 1. Detect invalid combos leading to zero features for selection
        fs_strategy = combo_dict.get('gs_features_selection_strategy', None)
        if (fs_strategy is not None
            and combo_dict.get('gs_use_band_features') == True
            and combo_dict.get('gs_use_glcm_features') == False
            and combo_dict.get('gs_use_other_features') == False
            and combo_dict.get('gs_fss_use_bands') == False):
            # We have only band features, but selection is ignoring bands => skip
            continue

        # -- 2. Detect invalid combos for PCA
        #    If apply_pca is True but we effectively skip the only active features,
        #    we have zero features for PCA => skip
        apply_pca = combo_dict.get('gs_apply_pca', False)
        pca_use_bands = combo_dict.get('gs_pca_use_bands', False)
        if (apply_pca
            and combo_dict.get('gs_use_band_features') == True
            and combo_dict.get('gs_use_glcm_features') == False
            and combo_dict.get('gs_use_other_features') == False
            and pca_use_bands == False):
            # Only band features exist, but PCA is ignoring bands => skip
            continue
        
        # -- 2.1. Dataset-specific exclusions for "dataset.sav"
        if is_dataset_sav:
            if combo_dict.get('gs_use_other_features') == True:
                continue
            if apply_pca and not pca_use_bands and combo_dict.get('gs_use_glcm_features') == False:
                continue
            
        # -- 3. Build a 'canonical' representation that eliminates differences in
        #       parameters that won't be used anyway.
        canonical_dict = combo_dict.copy()

        # If apply_pca is False, pca params are irrelevant -> set them to None
        if not canonical_dict.get('gs_apply_pca', True):
            canonical_dict['gs_pca_use_bands'] = None
            canonical_dict['gs_pca_components_selection_strategy'] = None

        # If features_selection_strategy is None, fss_use_bands is irrelevant -> set it to None
        if canonical_dict.get('gs_features_selection_strategy', None) is None:
            canonical_dict['gs_fss_use_bands'] = None

        # -- 4. Convert the canonical dict to a tuple in a consistent order, so we can
        #       store it in a set to avoid duplicates.
        canonical_tuple = tuple(sorted(canonical_dict.items()))
        if canonical_tuple not in seen:
            seen.add(canonical_tuple)
            filtered_combos.append(combo_dict)

    return filtered_combos


def grid_search(parameters: dict) -> None:
    """
    Executes a grid search over all valid configurations of the SVM pipeline.

    The function:
    - Parses 'gs_' parameters from the input dictionary and builds a cartesian product of options.
    - Prunes invalid and redundant combinations.
    - For each valid combination:
        - Loads and preprocesses the dataset.
        - Applies optional feature selection and PCA.
        - Trains both a multiclass SVM and binary SVMs.
        - Evaluates both uncalibrated and calibrated versions of the models.
        - Computes average F1-scores and saves all relevant metrics and parameters to a CSV.

    Parameters
    ----------
    parameters : dict
        Full configuration dictionary including both fixed and grid search parameters.
        Grid search values must be prefixed with 'gs_' and specified as lists.

    """
    
    parameters["verbose"] = False
    
    warnings.simplefilter("ignore", UserWarning)
    # To share variables
    
    grid_search_parameters = {k:v for k,v in parameters.items() if k.startswith('gs_')}
    parameters = {k:v for k,v in parameters.items() if not k.startswith('gs_')}
    
    print_dict(grid_search_parameters, title="Grid Search parameters values")
    combinations = list(itertools.product(*grid_search_parameters.values()))
    total_combinations = len(combinations)
    
    # Prune duplicates first:
    print(f"Combination before filtering: {total_combinations}")
    combinations = filter_grid_search_combinations(grid_search_parameters)
    total_combinations = len(combinations)
    print(f"Combination after filtering: {total_combinations}")

    grid_search_csv_path  = f"{parameters['output_path']}/grid_search.csv"
    not_to_store_params = ['fixed_features','perform_grid_search','output_path','verbose','log_dir','log_level','labels_path','features_used','features_not_used']

    try:
        df_existing_results = pd.read_csv(grid_search_csv_path)
    except FileNotFoundError:
        df_existing_results = pd.DataFrame()  # Create empty DataFrame if no results exist

    existing_combinations = {}
    if len(df_existing_results) > 0:

        for id, row in df_existing_results.iterrows():
            
            combo_id = row['id']            
            existing_combinations[combo_id] = {'SVM_AVGF1': row['SVM_AVGF1'], 'SVMS_AVGF1': row['SVMS_AVGF1']}
        
    print_dict(existing_combinations, title="\nExisting results initial")

    binary_svms_param_grid = {
        'C': np.linspace(100, 1000, 5),
        'gamma': np.linspace(1e-3, 10, 10),
        'probability' : [False]
    }
    
    mc_svm_param_grid = {
        'C': np.linspace(100, 1000, 5),
        'gamma': np.linspace(1e-3, 10, 10),
        'probability' : [True]
    }

    for i, combination in enumerate(combinations):
                
        try:
            current_params = parameters.copy()
            to_show = []
            
            print(combination)
            for key, val in combination.items():
                stripped_key = key.replace('gs_', '')
                to_show.append((key, val))
                current_params[stripped_key] = val
                
            current_params["dataset_type"] = current_params['dataset_path'].split('_')[-1][:-4]

            combo_tuple_as_str = '_'.join(map(str, combination.values()))
            curr_id = combo_tuple_as_str.lower()

            if curr_id in set(existing_combinations.keys()):
                print(f"\nCombination {i + 1} / {total_combinations} - id: {curr_id} - Already exist, skipping.")    
                print(f"    - SVM Avg. F1 : {existing_combinations[curr_id]['SVM_AVGF1']}")
                print(f"    - SVMs Avg. F1: {existing_combinations[curr_id]['SVMS_AVGF1']}")
                continue
            else:
                print(f"\nCombination {i + 1} / {total_combinations} - id: {curr_id}")            
            

            current_params['id'] = curr_id
            #print_dict(current_params, title="Current parameters")
            print_list(to_show, "   - Current settings:")
            print(f"   - Loading and preparing dataset")
            dataset = pd.read_csv(current_params["dataset_path"])
            dataset["ground_truth_index"] = dataset["ground_truth_index"].astype(int)

            current_params["non_features_columns"] = [f for f in current_params["non_features_columns"] if f in dataset.columns]
            
            dataset_train, dataset_test, dataset_calibration = split_dataset(dataset, current_params, generate_report=False)
            dataset_train, preprocessing_metadata = preprocess_features(dataset_train, current_params)        
                    
            scalers = {f : get_scaler(info["method"], info["params"]) for f, info in preprocessing_metadata.items()}

            for feature_name in dataset_train.keys():
                
                if feature_name in current_params["non_features_columns"] or feature_name in current_params["exluded_from_preprocessing"]:
                    continue
                
                dataset_test[feature_name] = scalers[feature_name].transform(dataset_test[feature_name].values.reshape(-1,1)).flatten()
                dataset_calibration[feature_name] = scalers[feature_name].transform(dataset_calibration[feature_name].values.reshape(-1,1)).flatten()


            features_to_remove, features_to_use, _ = features_selection(current_params, dataset_train)

            assert features_to_use.intersection(features_to_remove) == set(),  "Detected a feature to be removed and used at the same time."
            assert len(features_to_remove) + len(features_to_use) == len(dataset_train.columns),  f"Detected inconsistent number of dataset columns in features to use and to remove.\nFeatures to use: {len(features_to_use)}\nFeatures to remove: {len(features_to_remove)}"
            assert len(features_to_use) != len(current_params['non_features_columns']), "Parameters configuration error. Specify at least a feature to use."

            current_params["features_not_used"] = features_to_remove

            dataset_train = dataset_train.drop(columns=features_to_remove)[list(features_to_use)]
            dataset_calibration = dataset_calibration[list(dataset_train.keys())]
            dataset_test = dataset_test[list(dataset_train.keys())]
            
            if current_params["apply_pca"] == True:
                
                use_bands = parameters["pca_use_bands"] == True
                features_cols = [col for col in dataset_train.columns if col not in parameters['non_features_columns']]

                if not use_bands:
                    features_cols = [f for f in features_cols if not f.startswith('band_')]
                    
                if len(features_cols) != 0:
                    dataset_train, pca_metadata = apply_pca(dataset_train, current_params, plot=False)

                    dataset_test = reload_and_apply_PCA(pca_metadata, dataset_test, current_params)[dataset_train.columns]
                    dataset_calibration = reload_and_apply_PCA(pca_metadata, dataset_calibration, current_params)[dataset_train.columns]
                    
            Y_train = dataset_train["ground_truth_label"]
            Y_test = dataset_test["ground_truth_label"]
            Y_cal = dataset_calibration["ground_truth_label"]

            X_train = dataset_train.drop(columns=current_params["non_features_columns"])
            X_test = dataset_test.drop(columns=current_params["non_features_columns"])
            X_cal = dataset_calibration.drop(columns=current_params["non_features_columns"])

            features_names = X_train.columns
            current_params["features_used"] = features_names

            print(f"   - Features used: {len(current_params['features_used'])} (removed: {len(current_params['features_not_used'])})")
            
            print(f"   - Training Binary SVMs.")
            svms = train_binary_svms(current_params, binary_svms_param_grid, X_train, X_test, X_cal, Y_train, Y_test, Y_cal)
            
            print(f"   - Training Multiclass SVM.")
            svm_grid_search = train_svm(X_train, Y_train, mc_svm_param_grid, cv=3)
            svm = svm_grid_search.best_estimator_
            svm_report = svm_grid_search.best_params_
            svm_report = standardize_dict(svm_report, "SVM_params")
            svm_test_report = test_svm(svm, X_test, Y_test, verbose=current_params['verbose'])
            svm_test_report = standardize_dict(svm_test_report, "SVM_EVAL")
            svm_report.update(svm_test_report)
            svm_calibrated = CalibratedClassifierCV(estimator=svm, method='isotonic', cv='prefit', n_jobs=-1)
            svm_calibrated.fit(X_cal, Y_cal)
            svm_calibration_report = test_svm(svm_calibrated, X_test, Y_test, verbose=current_params['verbose'])
            svm_calibration_report = standardize_dict(svm_calibration_report, "SVM_EVALCAL")
            svm_report.update(svm_calibration_report)
            
            metrics = svm_report.copy()

            for k, (svm, svm_cal, report) in svms.items():
                        
                metrics.update(report.items())

            metrics.update({k:v for k,v in current_params.items() if not k in not_to_store_params})

            d_col_svm_f1 = [col for col in metrics.keys() if col.startswith('SVM_') and 'macro-avg_f1-score' in col]
            d_col_svms_f1 = [col for col in metrics.keys() if col.startswith('SVMS_') and 'macro-avg_f1-score' in col]

            svm_f1 = np.mean([metrics[k] for k in d_col_svm_f1])
            svms_f1 = np.mean([metrics[k] for k in d_col_svms_f1])
            
            existing_combinations[curr_id] = {'SVM_AVGF1': svm_f1, 'SVMS_AVGF1': svms_f1}
            
            metrics.update({"SVM_AVGF1":svm_f1})
            metrics.update({"SVMS_AVGF1":svms_f1})

            save_experiment_result(metrics, grid_search_csv_path)
            print(f"   - SVM Avg. F1 : {svm_f1}")
            print(f"   - SVMs Avg. F1: {svms_f1}")
            print(f"   - Results stored.")
            
        except Exception as e:
            print(f"   - Error in combination {i + 1} / {total_combinations} - id: {curr_id}")
            print(e)
            continue

