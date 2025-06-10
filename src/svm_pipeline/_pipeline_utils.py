import numpy as np
import csv
import os
from sklearn.decomposition import PCA
import pandas as pd

def standardize_dict(
    d_in: dict, 
    prefix: str
) -> dict:
    """
    Flattens a nested dictionary, appending a prefix to keys for standardization.

    Parameters
    ----------
    d_in : dict
        Input dictionary, potentially containing nested dictionaries.
    prefix : str
        Prefix for each key.

    Returns
    -------
    dict
        Flattened dictionary with standardized keys.
    """
    
    d_out = {}
    
    for key, value in d_in.items():
        if isinstance(value, dict):
            d_out.update({f"{prefix}_{key}_{k}".replace('.', '').replace(' ','-'):v for k, v in value.items()})
        else:
            d_out.update({f"{prefix}_{key}".replace('.', '').replace(' ','-') : value})
        
    return d_out


def create_next_experiment_folder(
    base_path: str
) -> str:
    """
    Creates a new experiment folder with an incremented index.

    Parameters
    ----------
    base_path : str
        Base directory for the experiment folders.

    Returns
    -------
    str
        Path of the newly created experiment folder.
    """
    experiment_dirs = [d for d in os.listdir(base_path) if d.startswith("experiment_") and d[11:].isdigit()]
    max_x = max((int(d[11:]) for d in experiment_dirs), default=-1)
    new_folder_name = f"experiment_{max_x + 1}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    return new_folder_path

            
def handle_value(
    value: any
) -> str:
    """
    Converts a value to a string suitable for CSV storage.

    Parameters
    ----------
    value : any
        The value to be processed (list, tuple, numpy array, float, string, or None).

    Returns
    -------
    str
        String representation of the value.
    """


    if isinstance(value, (list, tuple, np.ndarray, set)):
        return ','.join(map(str, value))  # Convert arrays and lists to comma-separated strings
    if value is None:
        return 'None'  # Represent None values as 'None'
    if isinstance(value, (float, np.float64)):
        return f"{value}" 
    return str(value)  # Convert everything else to a string

def save_experiment_result(data: dict, path: str) -> None:
    file_exists = os.path.exists(path)

    if file_exists:
        with open(path, 'r', newline='') as fr:
            reader = list(csv.reader(fr))
            old_header = reader[0]
            rows = reader[1:]
    else:
        old_header = ['index']
        rows = []

    data_keys = sorted(list(data.keys()))
    updated_header = list(old_header)

    # Add new keys to the header
    for key in data_keys:
        if key not in updated_header:
            updated_header.append(key)

    header_changed = updated_header != old_header

    # Rewrite file with updated header if needed
    if header_changed:
        # Update existing rows with None for missing fields
        updated_rows = []
        for row in rows:
            row_dict = dict(zip(old_header, row))
            new_row = [row_dict.get(col, 'None') for col in updated_header]
            updated_rows.append(new_row)

        with open(path, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(updated_header)
            writer.writerows(updated_rows)

    # Determine next index
    if rows:
        max_index = max(int(row[0]) for row in rows if row and row[0].isdigit())
    else:
        max_index = -1

    new_index = max_index + 1

    # Build new row
    new_row = []
    for key in updated_header:
        if key == 'index':
            new_row.append(str(new_index))
        else:
            new_row.append(handle_value(data.get(key)))

    # Append new row
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and not header_changed:
            writer.writerow(updated_header)
        writer.writerow(new_row)



def reload_and_apply_PCA(pca_metadata, dataset, parameters):
    # Restore PCA model
    pca_model = PCA()
    pca_model.components_ = pca_metadata["params"]["components_"]
    pca_model.n_components_ = pca_metadata["params"]["n_components_"]
    pca_model.explained_variance_ = pca_metadata["params"]["explained_variance_"]
    pca_model.singular_values_ = pca_metadata["params"]["singular_values_"]
    pca_model.mean_ = pca_metadata["params"]["mean_"]
    pca_model.n_samples_ = pca_metadata["params"]["n_samples_"]
    pca_model.noise_variance_ = pca_metadata["params"]["noise_variance_"]
    pca_model.n_features_in_ = pca_metadata["params"]["n_features_in_"]
    pca_model.feature_names_in_ = pca_metadata["params"]["feature_names_in_"]
    
    # Extract different column groups
    non_feature_cols = dataset[parameters["non_features_columns"]]
    pca_input_features = dataset[pca_model.feature_names_in_]

    # All feature columns (i.e., dataset minus non_features)
    all_features = dataset.drop(columns=parameters["non_features_columns"])
    
    # Identify non-PCA features (remaining features not used in PCA)
    non_pca_feature_cols = all_features.drop(columns=pca_model.feature_names_in_)

    # Apply PCA
    transformed_data = pca_model.transform(pca_input_features)
    pca_columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
    df_pca = pd.DataFrame(transformed_data, columns=pca_columns, index=dataset.index)
    
    # Combine everything
    result = pd.concat([non_feature_cols, non_pca_feature_cols, df_pca], axis=1)
    
    return result
