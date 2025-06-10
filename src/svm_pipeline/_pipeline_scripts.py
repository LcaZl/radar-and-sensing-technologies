import logging
import pandas as pd
import numpy as np
import os
from sklearn.calibration import CalibratedClassifierCV

from scripting import logged_main, print_dict
from ._features_selection import features_selection, store_ma_output, apply_pca, store_rfecv_output, correlation_analysis, features_importance_analysis
from ._features_preprocessing import preprocess_features, split_dataset, get_scaler
from ._svms_evaluation import ( train_svm, test_svm, train_binary_svms, SVMS_prediction_test)
from ._svms import save_SVMS, SVMS
from ._svms_calibration_curves import multiclass_svm_calibration_curves, binary_svms_calibration_curves
from ._pipeline_utils import standardize_dict, create_next_experiment_folder, reload_and_apply_PCA
from ._svms_grid_search import grid_search

def pipeline(**parameters) -> None:
        
    logger = logging.getLogger("svm-tr-pipeline")    

    if parameters['perform_grid_search'] == True:
        logger.info("Grid search for SVM pipeline settings")
        
        grid_search(parameters)
        return
    
    logger.info("SVMs pipeline - Train SVM")

    # Setup experiment parameters
    
    parameters = {k:v for k,v in parameters.items() if not k.startswith("GridSearch")}
    
    os.makedirs(parameters['output_path'], exist_ok=True)
    experiment_path = create_next_experiment_folder(parameters["output_path"])
    
    parameters["dataset_type"] = parameters['dataset_path'].split('_')[-1][:-4]
    parameters["features_inspection_path"] = f"{experiment_path}/features_info"
    parameters["svm_model_path"] = f"{experiment_path}/svm.pkl"
    parameters["calibrated_model_path"] = f"{experiment_path}/svm_calibrated.pkl"
    parameters["calibration_curves_path"] = f"{experiment_path}/calibration_curves"
    parameters["performance_csv_path"] = f"{parameters['output_path']}/experiments.csv"    # Load dataset and create labes vocabulary

    os.makedirs(parameters["features_inspection_path"], exist_ok=True)
    os.makedirs(parameters["calibration_curves_path"], exist_ok=True)

    # Loading dataset
    
    dataset = pd.read_csv(parameters["dataset_path"])
    dataset["ground_truth_index"].astype(int)
    parameters["non_features_columns"] = [f for f in parameters["non_features_columns"] if f in dataset.columns]
        
    # Features preprocessing and selection
    
    logger.info("Preprocessing features ...")   
    dataset_train, dataset_test, dataset_calibration = split_dataset(dataset, parameters, generate_report=False)
    dataset_train, preprocessing_metadata = preprocess_features(dataset_train, parameters)
    scalers = {f : get_scaler(info["method"], info["params"]) for f, info in preprocessing_metadata.items()}

    for feature_name in dataset_train.keys():
        
        if feature_name in parameters["non_features_columns"] or feature_name in parameters["exluded_from_preprocessing"]:
            continue
        
        dataset_test[feature_name] = scalers[feature_name].transform(dataset_test[feature_name].values.reshape(-1,1)).flatten()
        dataset_calibration[feature_name] = scalers[feature_name].transform(dataset_calibration[feature_name].values.reshape(-1,1)).flatten()
    print("\n\n")
                
    features_to_remove, features_to_use, selection_report = features_selection(parameters, dataset_train)

    assert features_to_use.intersection(features_to_remove) == set(),  "Detected a feature to be removed and used at the same time."
    assert len(features_to_remove) + len(features_to_use) == len(dataset_train.columns),  f"Detected inconsistent number of dataset columns in features to use and to remove.\nFeatures to use: {len(features_to_use)}\nFeatures to remove: {len(features_to_remove)}"

    print(f"\nLV. 0 - Total features to analyze: {len(dataset_train.columns)} ({len(parameters['non_features_columns'])} Non Features)")
    print(f"LV. 1 - Features intially removed: {len(selection_report['removed_features_lv1'])}")
    print(f"LV. 2 - Removed constant Features: {len(selection_report['constant_features'])}")
    print(f"LV. 3 - Features removed with {parameters['features_selection_strategy']} selection strategy: {len(selection_report['removed_features_lv2'])}")
    print(f"LV. 4 - Features to be used: {len(features_to_use)} ({len(parameters['non_features_columns'])} Non Features)")

    if len(features_to_use) == len(parameters['non_features_columns']):
        raise ValueError("Parameters configuration error. Specify at least a feature to use.")


    if selection_report["type"] == "multicollinearity_analysis" and parameters["verbose"] == True:
        
        correlation_matrix = selection_report["info"]["correlation_matrix"]
        features_importances = selection_report["info"]["features_importance"]
        store_ma_output(correlation_matrix, features_importances, parameters, note = "pre_selection")

    elif selection_report["type"] == "rfecv" and parameters["verbose"] == True:
        
        correlation_matrix = correlation_analysis(dataset_train, parameters)
        features_importances = features_importance_analysis(dataset_train, parameters, target_col='ground_truth_index')
        store_ma_output(correlation_matrix, features_importances, parameters, note = "pre_selection")

        store_rfecv_output(selection_report, parameters, note="")

    else:
        
        correlation_matrix = correlation_analysis(dataset_train, parameters)
        features_importances = features_importance_analysis(dataset_train, parameters, target_col='ground_truth_index')
        store_ma_output(correlation_matrix, features_importances, parameters, note = "pre_selection")


    dataset_train = dataset_train.drop(columns=features_to_remove)
    dataset_calibration = dataset_calibration[list(dataset_train.keys())]
    dataset_test = dataset_test[list(dataset_train.keys())]

    correlation_matrix = correlation_analysis(dataset_train, parameters)
    features_importances = features_importance_analysis(dataset_train, parameters, target_col='ground_truth_index')
    store_ma_output(correlation_matrix, features_importances, parameters, note = "post_selection")

    parameters["features_not_used"] = features_to_remove
    features_names = dataset_train.columns
    parameters["features_used"] = features_names
    
    pca = None
    if parameters["apply_pca"] == True:
                    
        dataset_train, pca = apply_pca(dataset_train, parameters)
        
        correlation_matrix = correlation_analysis(dataset_train, parameters)
        features_importances = features_importance_analysis(dataset_train, parameters, target_col='ground_truth_index')
        store_ma_output(correlation_matrix, features_importances, parameters, note = "post_pca")
    
        dataset_test = reload_and_apply_PCA(pca, dataset_test, parameters)[dataset_train.columns]
        dataset_calibration = reload_and_apply_PCA(pca, dataset_calibration, parameters)[dataset_train.columns]
        
    
    Y_train = dataset_train["ground_truth_label"]
    Y_test = dataset_test["ground_truth_label"]
    Y_cal = dataset_calibration["ground_truth_label"]

    X_train = dataset_train.drop(columns=parameters["non_features_columns"])
    X_test = dataset_test.drop(columns=parameters["non_features_columns"])
    X_cal = dataset_calibration.drop(columns=parameters["non_features_columns"])


    if parameters["verbose"]:
        logger.info(f"Shapes of prepared dataset:")
        print(f"    Train X           : {X_train.shape}")
        print(f"    Train Y           : {Y_train.shape}")
        print(f"    Test  X           : {X_test.shape}")
        print(f"    Test  Y           : {Y_test.shape}")
        print(f"    Cal.  X           : {X_cal.shape}")
        print(f"    Cal.  Y           : {Y_cal.shape}")

    logger.info("Dataset prepared")
    
    # SVM grid search parameters
    param_grid = {
        'C': np.linspace(100, 1000, 5),
        'gamma': np.linspace(1e-3, 10, 10),
        'class_weight': ['balanced', None],
        'random_state' : [42],
        'kernel' : ['rbf'],
        'probability' : [False]
    }
    
    logger.info("Training binary SVMs with parameters grid:")
    print(param_grid)
    svms = train_binary_svms(parameters, param_grid, X_train, X_test, X_cal, Y_train, Y_test, Y_cal)

    param_grid = {
        'C': np.linspace(100, 1000, 5),
        'gamma': np.linspace(1e-3, 10, 10),
        'class_weight': ['balanced', None],
        'random_state' : [42],
        'kernel' : ['rbf'],
        'probability' : [True]
    }
    
    logger.info("Train Multiclass SVM with parameters grid:")
    print(param_grid)
    svm_grid_search = train_svm(X_train, Y_train, param_grid, cv=3)
    
    svm = svm_grid_search.best_estimator_
    best_params = svm_grid_search.best_params_
    best_params_std = standardize_dict(best_params, "SVM_params")

    svm_test_report = test_svm(svm, X_test, Y_test, verbose=True)
    svm_test_report = standardize_dict(svm_test_report, "SVM_EVAL")
    svm_test_report.update(best_params_std)
    
    svm_calibrated = CalibratedClassifierCV(estimator=svm, method='isotonic', cv='prefit', n_jobs=-1)
    svm_calibrated.fit(X_cal, Y_cal)

    svm_calibration_report = test_svm(svm_calibrated, X_test, Y_test, verbose=True)
    svm_calibration_report = standardize_dict(svm_calibration_report, "SVM_EVALCAL")
    
    svm_test_report.update(svm_calibration_report)
    mcsvm_report = svm_test_report.copy()
         
    # Create SVMS class for all the svm trained
    classes = np.unique(Y_train)
    model = SVMS(svm=svm,
                svm_calibrated=svm_calibrated,
                svms=svms, 
                parameters=parameters, 
                classes=classes, 
                preprocessing_metadata=preprocessing_metadata, 
                training_report= mcsvm_report,  
                pca_metadata=pca, 
                features_used = features_names) 
    
    model.verbose = True
    save_SVMS(model, parameters['svm_model_path'])

    # Save calibration curves
    multiclass_svm_calibration_curves(model, X_test, Y_test, output_dir=parameters['calibration_curves_path'], show_plots=True)
    binary_svms_calibration_curves(model, X_test, Y_test, parameters['calibration_curves_path'], show_plots=True)


    # TESTS
    
    
    # Multiclass SVM test
    logger.info("Multiclass SVM evaluation")
    model.use_binary_svms = False
    model.use_multiclass_svm_calibrated = False
    model.use_binary_svms_with_softmax = False
    SVMS_prediction_test(model, X_test, Y_test, n_samples=None)
    model.verbose = False
    _ = test_svm(model, X_test, Y_test, verbose=True)

    # Multiclass SVM calibrated test
    logger.info("Multiclass SVM calibrated evaluation")
    model.use_binary_svms = False
    model.use_multiclass_svm_calibrated = True
    model.use_binary_svms_with_softmax = False

    SVMS_prediction_test(model, X_test, Y_test, n_samples=None)
    model.verbose = False
    _ = test_svm(model, X_test, Y_test, verbose=True)

    # Binary SVMs test
    logger.info("Binary SVMs evaluation")
    model.use_binary_svms = True
    model.use_multiclass_svm_calibrated = False
    model.use_binary_svms_with_softmax = False
    
    SVMS_prediction_test(model, X_test, Y_test, n_samples=None)
    model.verbose = False
    _ = test_svm(model, X_test, Y_test, verbose=True)

    # Binary SVMs with softmax test
    logger.info("Binary SVMs with Softmax evaluation")
    model.use_binary_svms = True
    model.use_multiclass_svm_calibrated = False
    model.use_binary_svms_with_softmax = True
    
    SVMS_prediction_test(model, X_test, Y_test, n_samples=None)
    model.verbose = False
    _ = test_svm(model, X_test, Y_test, verbose=True)


def pipeline_script() -> None:
    logged_main(
        "Train or Evaluate SVMs",
        pipeline
    )
    
