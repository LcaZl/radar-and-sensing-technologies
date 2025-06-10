import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from generate_dataset import inspect_class_distribution

class MinMaxScalerWithQuantiles():
    """
    Custom MinMaxScaler that uses quantiles for scaling.
    This is useful for datasets with outliers or non-normal distributions.
    """

    def __init__(self, min_q_val=0.001, max_q_val=0.999, scale = None, q_min = None, q_max = None):
        self.q_min_val = min_q_val
        self.q_max_val = max_q_val
        self.scale = scale
        self.q_min = q_min
        self.q_max = q_max
        
    def transform(self, X):
        if self.scale is None:
            raise ValueError("The scaler has not been fitted yet. Please call fit_transform first.")
        
        normalized_feature = (X - self.q_min) / (self.q_max - self.q_min)
        normalized_feature[normalized_feature < 0] = 0
        normalized_feature[normalized_feature > 1] = 1
        return normalized_feature
    
    def fit_transform(self, X):
        # Quantile-based min-max scaling
        self.q_min = np.quantile(X, self.q_min_val)
        self.q_max = np.quantile(X, self.q_max_val)
        self.scale = self.q_max - self.q_min

        if self.scale == 0:  
            self.scale = 1
        normalized_feature = (X - self.q_min) / self.scale
        normalized_feature[normalized_feature < 0] = 0
        normalized_feature[normalized_feature > 1] = 1
        
        return normalized_feature   
     
class DegreeToSinCos:
    """
    Transforms angular features (in degrees) into their sine and cosine components.
    This transformation preserves cyclic properties, making it suitable for circular features like aspects.

    Methods
    -------
    fit_transform(array)
        Transforms the input array into sine and cosine components.
    transform(array)
        Transforms the input array into sine and cosine components.
    """
    
    def __init__(self):
        pass
    
    def fit_transform(self, array):
        sin = np.sin(np.deg2rad(array))
        cos = np.cos(np.deg2rad(array))
        return sin, cos
    
    def transform(self, array):
        sin = np.sin(np.deg2rad(array))
        cos = np.cos(np.deg2rad(array))
        return sin, cos

def preprocess_features(
    dataset: pd.DataFrame, 
    parameters: dict
) -> tuple:
    """
    Preprocesses features in the dataset by applying scaling or transformations 
    specified in the parameters.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing features to preprocess.
    parameters : dict
        Configuration parameters including:
        - "other_features_preprocessing": Normalization method for non-band features.
        - "band_features_preprocessing": Normalization method for spectral bands.
        - "non_features_columns": Columns to exclude from preprocessing.
        - "verbose": If True, print statistics before and after transformations.

    Returns
    -------
    tuple
        - pd.DataFrame: Normalized features as a DataFrame.
        - dict: Metadata about the preprocessing methods and parameters to be stored.
    """

    logger = logging.getLogger("features-preprocessing")
    
    feature_preprocessing_method = parameters["other_features_preprocessing"]
    bands_preprocessing_method = parameters["band_features_preprocessing"]
    normalized_features = {}
    feature_metadata = {}

    for feature_name in dataset.columns:
        if feature_name in parameters["non_features_columns"] or feature_name in parameters["exluded_from_preprocessing"]:
            normalized_features[feature_name] = dataset[feature_name]
            continue

        feature = dataset[feature_name].values.reshape(-1, 1)
        original_values = feature.copy()
        current_preprocessing = bands_preprocessing_method if feature_name.startswith('band_') else feature_preprocessing_method

        if current_preprocessing == 'standard':
            scaler = StandardScaler()
            normalized_feature = scaler.fit_transform(original_values)
            normalized_features[feature_name] = normalized_feature.flatten()

            feature_metadata[feature_name] = {
                "method": 'standard',
                "params": {
                    "scale_": scaler.scale_,
                    "mean_": scaler.mean_,
                    "var_": scaler.var_,
                    "n_features_in_": scaler.n_features_in_,
                    "n_samples_seen_": scaler.n_samples_seen_
                }
            }

        elif current_preprocessing == 'minmax_q':
            
            scaler = MinMaxScalerWithQuantiles()
            normalized_feature = scaler.fit_transform(original_values)
            normalized_features[feature_name] = normalized_feature.flatten()

            feature_metadata[feature_name] = {
                "method": 'minmax_q',
                "params": {
                    "q_min": float(scaler.q_min),
                    "q_max": float(scaler.q_max),
                    "scale": float(scaler.scale)
                }
            }

        elif current_preprocessing == 'minmax':
            
            scaler = MinMaxScaler()
            normalized_feature = scaler.fit_transform(original_values)
            normalized_features[feature_name] = normalized_feature.flatten()

            feature_metadata[feature_name] = {
                "method": 'minmax',
                "params": {"min_" : scaler.min_,
                           "scale_" : scaler.scale_,
                           "data_min_" : scaler.data_max_,
                           "data_max_" : scaler.data_min_,
                           "data_range_" : scaler.data_range_,
                           "n_features_in_" : scaler.n_features_in_,
                           "n_samples_seen_" : scaler.n_samples_seen_
                           },
            }
            
        else:
            raise ValueError(f"Unknown preprocessing method: {current_preprocessing}")

        if parameters["verbose"]:
            logger.info(f"Feature: {feature_name} - {current_preprocessing}")
            print(f"  Original Statistics -> Min: {feature.min():.3f}, Max: {feature.max():.3f}, Mean: {feature.mean():.3f}, Std: {feature.std():.3f}, Var: {feature.var():.3f}")
            print(f"  Preprocessed Statistics -> Min: {normalized_feature.min():.3f}, Max: {normalized_feature.max():.3f}, Mean: {normalized_feature.mean():.3f}, Std: {normalized_feature.std():.3f}, Var: {normalized_feature.var():.3f}")

    normalized_df = pd.DataFrame(normalized_features)
    normalized_df = normalized_df[dataset.columns]  # Preserve original column order

    return normalized_df, feature_metadata

def split_dataset(
    dataset: pd.DataFrame, 
    parameters: dict, 
    generate_report: bool = True
) -> tuple:
    """
    Splits the dataset into training, testing and calibration sets.
    Optionally generates class distribution reports for each subset.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the dataset with class labels.
    parameters : dict
        Configuration parameters including:
        - "dataset_type": Type of dataset for custom splitting logic.
        - "non_features_columns": Columns to exclude from features.
        - "features_inspection_path": Path to save class distribution reports.
        - "verbose": If True, plots class distribution report.
    generate_report : bool, optional
        If True, generates, store and eventually show class distribution reports.

    Returns
    -------
    tuple
        - X_train, X_test, X_cal : pd.DataFrame
            Training, testing, and calibration feature sets.
        - Y_train, Y_test, Y_cal : pd.Series
            Corresponding labels for training, testing, and calibration sets.
    """
     
    #logger = logging.getLogger("split-dataset")
    dataset_type = parameters['dataset_type']

    # Split the dataset into train, test and calibration
    
    if dataset_type == "fullLC":
        
        # Dataset to divide with split columns
        #   - Train, samples from LC map only (split == train)
        #   - Test/Cal, samples from shape file (split == test)
        
        dataset_train = dataset[dataset["split"] == "train"]
        dataset_test = dataset[dataset["split"] == "test"]
        dataset_test, dataset_cal = train_test_split(dataset_test, test_size=0.5, stratify=dataset_test["ground_truth_index"])
    
    else:
        
        # Dataset without split column.
        # It can be made of:
        #   - samples from shape file only
        #   - samples from shape file + LC map (enhanced)
        
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, stratify=dataset['ground_truth_index'])
        dataset_test, dataset_cal = train_test_split(dataset_test, test_size=0.5, stratify=dataset_test["ground_truth_index"])
        
    if generate_report:
        
        train_cls_distr = inspect_class_distribution(dataset_train, 
            class_column='ground_truth_label', 
            title = f"Train dataset Class Distribution for {len(dataset_train)} Points.",
            plot=parameters["verbose"])
        
        test_cls_distr = inspect_class_distribution(dataset_test,
            class_column='ground_truth_label', 
            title = f"Test dataset Class Distribution for {len(dataset_test)} Points.",
            plot=parameters["verbose"])
    
        
        cal_cls_distr = inspect_class_distribution(dataset_cal, 
            class_column='ground_truth_label', 
            title = f"Calibration dataset Class Distribution for {len(dataset_cal)} Points.",
            plot=parameters["verbose"])
        
        train_class_distribution_path = f"{parameters['features_inspection_path']}/train_class_distribution.png"
        test_class_distribution_path = f"{parameters['features_inspection_path']}/test_class_distribution.png"
        cal_class_distribution_path = f"{parameters['features_inspection_path']}/cal_class_distribution.png"

        train_cls_distr.savefig(train_class_distribution_path)
        test_cls_distr.savefig(test_class_distribution_path)
        cal_cls_distr.savefig(cal_class_distribution_path)
        
    return dataset_train, dataset_test, dataset_cal

def get_scaler(
    scaler_type: str, 
    args: dict = {}
):
    """
    Returns a scaler object based on the specified scaling type and provided parameters.

    Parameters
    ----------
    scaler_type : str
        Type of scaler to return. Supported values:
        - "standard": StandardScaler with optional mean, var and scale parameters.
        - "minmax": MinMaxScaler with optional min, max, scale and range parameters.
        - "degree_to_sincos" or "degree2sincos": DegreeToSinCos for angular features.
    args : dict, optional
        Dictionary of parameters for initializing the scaler.

    Returns
    -------
    StandardScaler | MinMaxScaler | DegreeToSinCos
        The corresponding scaler object.

    Raises
    ------
    ValueError
        If an unsupported scaling method is requested.
    """

    if scaler_type == "standard":
        if len(args) > 0:

            scaler = StandardScaler()            
            scaler.scale_ = args["scale_"][0]
            scaler.mean_ = args["mean_"][0]
            scaler.var_ = args["var_"][0]
            scaler.n_features_in_ = args["n_features_in_"]
            scaler.n_samples_seen_ = args["n_samples_seen_"]
            return scaler
        else:
            return StandardScaler()

    elif scaler_type == "minmax":
        if len(args) > 0:

            scaler = MinMaxScaler(feature_range=(args["data_min_"][0], args["data_max_"][0]))
            scaler.min_ = args["min_"][0]
            scaler.scale_ = args["scale_"][0]
            scaler.data_min_ = args["data_min_"][0]
            scaler.data_max_ = args["data_max_"][0]
            scaler.data_range_ = args["data_range_"][0]
            scaler.n_features_in_ = args["n_features_in_"]
            scaler.n_samples_seen_ = args["n_samples_seen_"]
            
            return scaler

        else:
            return MinMaxScaler()

    elif scaler_type == "minmax_q":
        if len(args) > 0:
            scaler = MinMaxScalerWithQuantiles(q_max=args["q_max"], q_min=args["q_min"], scale=args["scale"])
            return scaler
        else:
            return MinMaxScalerWithQuantiles()
        
    else:
        raise ValueError(f"Requested scaling method {scaler_type} not supported. "
                         "Available: standard, minmax, minmax_q.")
