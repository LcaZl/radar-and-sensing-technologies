import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from kneed import KneeLocator

from scripting import print_dataframe

def features_selection(
    parameters: dict, 
    dataset: pd.DataFrame
) -> tuple:
    """
    Performs feature selection on the input dataset based on the specified parameters.
    It includes filtering of band, GLCM and other features, as well as optional 
    constant feature removal and other selection strategies like multicollinearity analysis and RFECV.

    Parameters
    ----------
    parameters : dict
        Configuration dictionary containing:
        - "non_features_columns": List of columns to exclude from feature selection.
        - "fixed_features": List of features that must always be retained, despite other parameters.
        - "use_band_features": Boolean flag to include band features.
        - "use_glcm_features": Boolean flag to include GLCM features.
        - "use_other_features": Boolean flag to include other features.
        - "drop_constant_features": Boolean flag to drop constant features.
        - "features_selection_strategy": Strategy for advanced feature selection:
            - "multicollinearity_analysis"
            - "rfecv"
    dataset : pd.DataFrame
        Input dataset containing features and non-feature columns.

    Returns
    -------
    tuple
        - features_removed (set): Set of features removed during selection.
        - features_to_use (set): Set of features retained.
        - features_analysis (dict): Detailed analysis of feature selection including:
            - "type": Feature selection strategy applied.
            - "removed_features_lv1": Features removed in the initial filtering.
            - "constant_features": Features removed due to constant values.
            - "removed_features_lv2": Features removed during advanced selection.
            - "info": Additional information from advanced selection methods.
            - "features_to_use": Final set of features to keep.
            - "features_removed": Final set of all removed features.
    """
    logger = logging.getLogger("dataset-preparation")
    
    # Setup features sets
    
    features_removed = set()
    features_to_use = set()
    
    non_features_columns = set(f for f in parameters["non_features_columns"] if f in dataset.columns) 
    glcm_features_columns = {f for f in dataset.columns if f.startswith("GLCM")}
    band_features_columns = {f for f in dataset.columns if f.startswith("band")}
    other_features_columns = {f for f in dataset.columns if
                                not f.startswith("band") and
                                not f.startswith("GLCM") and
                                not f in non_features_columns}
    
    # Check that the fixed features eventually requested are available
    fixed_features = set(parameters["fixed_features"])
    for f in fixed_features:
        if f not in dataset.columns:
            raise ValueError(f"Fixed feature: {f} not in dataset. Correct or remove this fixed feature.")
    
    # Identify all potential features
    candidate_features = set()
    
    if parameters["use_band_features"] == True: # -> Bands will be used
        candidate_features.update(band_features_columns)
    else:
        features_removed.update(band_features_columns)
        
    if parameters["use_glcm_features"] == True: # -> GLCM features will be used
        candidate_features.update(glcm_features_columns)
    else:
        features_removed.update(glcm_features_columns)
         
    if parameters["use_other_features"] == True: # -> All the non bands and non GLCM features will be used
        candidate_features.update(other_features_columns)
    else:
        features_removed.update(other_features_columns)
        
    candidate_features.difference_update(non_features_columns) # Remove non features columns from candidates
    
    # Add the fixed features (if already inside, the set manage this)
    candidate_features.update(fixed_features)
    
    # Remove from the removed features the fixed ones, they are always used.
    features_removed.difference_update(fixed_features)

    # If a fixed features is constant will be removed.    
    if parameters["drop_constant_features"] == True:
        
        constant_features = set()
        for f in candidate_features:
            min_val, max_val = dataset[f].min(), dataset[f].max()
            if min_val == max_val:
                constant_features.add(f)
                
        candidate_features.difference_update(constant_features)
        features_removed.update(constant_features)

    ma_dataset = dataset.drop(columns=features_removed)

    features_analysis = {
        "type" : parameters["features_selection_strategy"],
        "removed_features_lv1" : features_removed.copy(),
        "constant_features" : constant_features.copy()
    }
    
    if parameters["features_selection_strategy"] ==  "multicollinearity_analysis":
        
        # IDentifying columns to remove
        ma_to_remove, info = multicollinearity_analysis(ma_dataset, parameters=parameters)

        candidate_features.difference_update(set(ma_to_remove))
        features_removed.update(set(ma_to_remove))
        
        features_analysis.update({
            "removed_features_lv2" : ma_to_remove,
            "info" : info
        })
        
    elif parameters["features_selection_strategy"] ==  "rfecv":
        
        rfe_to_remove = []
        info = {}
        
        if len([f for f in ma_dataset.columns if not f in non_features_columns]) == 1:
            logger.info(f"RFECV analysis not possible with one feature, skipping.")
        else:
            rfe_to_remove, info = rfecv_analysis(ma_dataset, parameters)
        
        candidate_features.difference_update(set(rfe_to_remove))
        features_removed.update(set(rfe_to_remove))
        
        features_analysis.update({
            "removed_features_lv2" : rfe_to_remove,
            "info" : info
        })
    else:
        features_analysis.update({
            "removed_features_lv2" : [],
            "info" : None
        })
        
    # Remaining features to use
    features_to_use = candidate_features.union(non_features_columns)
    
    features_analysis.update({
        "features_to_use" : features_to_use,
        "features_removed" : features_removed
    })

    return features_removed, features_to_use, features_analysis


def rfecv_analysis(
    ma_dataset: pd.DataFrame, 
    parameters: dict
) -> tuple:
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select optimal features.

    Parameters
    ----------
    ma_dataset : pd.DataFrame
        DataFrame containing features and target labels.
    parameters : dict
        Configuration parameters including:
        - "non_features_columns": Columns to exclude from feature selection.
        - "fss_use_bands": Boolean flag to include band features in the analysis.

    Returns
    -------
    tuple
        - removed_features (list): List of features removed by RFECV.
        - info (dict): Information about RFECV results, including:
            - "rfecv": RFECV model object.
            - "features_names_in": List of input feature names.
            - "selected_features": List of features selected by RFECV.
            - "n_selected_features": Number of selected features.
    """

    logger = logging.getLogger("rfecv-features-selection")
    
    # Setup features to use
    use_bands = parameters["fss_use_bands"] == True    
    feature_cols = [col for col in ma_dataset.columns if not col in parameters['non_features_columns']]
    band_features_cols = [f for f in ma_dataset.columns if f.startswith('band_')]

    if not use_bands:
        feature_cols = [f for f in feature_cols if not f.startswith('band_')]
    
    feature_cols = [f for f in feature_cols if not f in parameters["fixed_features"]]
    
        
    # Select the identified features
    X = ma_dataset[feature_cols]
    y = ma_dataset['ground_truth_index']

   # Define the base SVM model
    svm = SVC(kernel='linear', probability=True)
    
    if parameters["verbose"] == True:
        logger.info("Evaluating features with RFECV ...")
    rfecv = RFECV(estimator=svm, step=1, cv=StratifiedKFold(3), scoring="f1_macro", n_jobs=-1)
    rfecv.fit_transform(X, y)

    # Create removed and to use features sets
    removed_features = []
    selected_features = []
    for feature, to_use in zip(feature_cols, rfecv.support_):
        if to_use:
            selected_features.append(feature)
        else:
            removed_features.append(feature)
    
    # If bands were not involved in the analysis, add them back to the features selected, as well as the fixed features.
    if not use_bands:
        for f in band_features_cols:
            selected_features.append(f)
            
    for f in parameters["fixed_features"]:
        selected_features.append(f)
            
    info = {
        "rfecv": rfecv,
        "features_names_in" : feature_cols,
        "selected_features": selected_features,
        "n_selected_features": rfecv.n_features_
    }
        
    return removed_features, info


def multicollinearity_analysis(
    dataset: pd.DataFrame, 
    parameters: dict
) -> tuple:
    """
    Analyzes multicollinearity among features and identifies redundant features to drop 
    using a combination of correlation analysis and feature importance metrics.

    The selection logic follows these steps:
    1. Compute the correlation matrix for all features.
    2. Calculate the mean correlation for each feature to get the overall correlation strength.
    3. Evaluate feature importance using mutual information scores.
    4. Combine importance scores with mean correlation to compute a combined metric:
       - combined_metric = importance / (1 + |mean_correlation|)
    5. Rank features by the combined metric to prioritize relevant features.
    6. Iterate through highly correlated feature pairs:
       - For each pair, compare combined metrics.
       - Drop the feature with the lower combined metric to minimize information loss.
    7. Ensure fixed features are retained regardless of correlation.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the input features.
    parameters : dict
        Configuration parameters including:
        - "ma_correlation_threshold": Correlation threshold for dropping features.
        - "non_features_columns": List of columns to exclude from analysis.
        - "fixed_features": List of features to always retain.
        - "verbose": If True, prints detailed logs and matrices.

    Returns
    -------
    tuple
        - to_drop (list): List of features identified as redundant and to be dropped.
        - info (dict): Detailed analysis information including:
            - "correlation_matrix": Correlation matrix of features.
            - "features_importance": Feature importance scores.
    """

    
    logger = logging.getLogger("multicollinearity-analysis")
    
    if parameters["verbose"] == True:
        logger.info(f"Multicollinearity analysis:\n")
    threshold = parameters["ma_correlation_threshold"]
    
    corr_matrix = correlation_analysis(dataset, parameters)
    importance_df = features_importance_analysis(dataset, parameters, target_col='ground_truth_index')
    corr_matrix_mean = corr_matrix.mean(axis = 1).reset_index()

    to_drop = []
    
    corr_matrix_mean.columns = ['feature', 'mean_correlation']
    metrics = pd.merge(corr_matrix_mean, importance_df, on='feature', how='inner')
    
    metrics['combined_metric'] = metrics['importance'] / (1 + metrics['mean_correlation'].abs())
    
    # Sort by importance descending
    metrics = metrics.sort_values(by='combined_metric', ascending=False).reset_index(drop=True)
    metrics = metrics.set_index("feature")

    # The matrix is symmetric. Set diagonal and below values to nan.
    # Then work col by col considering only the non nan combos.
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    if parameters["verbose"] == True:
        print_dataframe(upper, title="Correlation Matrix for pairs comparison - Processed by columns")
    
    for feat_A in upper.columns:
        
        # Identify features pair with A fixed which correlation is above the threshold
        high_corr_features = upper.index[abs(upper[feat_A]) > threshold].tolist()
        
        if parameters["verbose"] == True:
            logger.info(f"Current (A) Feature: {feat_A} - High correlated features: {high_corr_features}")
        
        for feat_B in high_corr_features:
            
            score_A = metrics.loc[feat_A].combined_metric #importance_dict.get(col, 0)
            score_B = metrics.loc[feat_B].combined_metric #importance_dict.get(feat, 0)
            
            if parameters["verbose"] == True:
                logger.info(f"Combination: (A) {feat_A} - (B) {feat_B}:\n- Scores A/B: {score_A}/{score_B}")
            
            if score_A > score_B:
                if feat_B not in to_drop:
                    to_drop.append(feat_B)
                    if parameters["verbose"] == True:
                        logger.info(f"- Dropped: {feat_B}")
            else:
                if feat_A not in to_drop:
                    to_drop.append(feat_A)
                    if parameters["verbose"] == True:
                        logger.info(f"- Dropped: {feat_A}")
                    break
                
    info = {
        "correlation_matrix" : corr_matrix,
        "features_importance" : importance_df
    }
    
    return list(set(to_drop)), info


def features_importance_analysis(
    df: pd.DataFrame, 
    parameters: dict, 
    target_col: str, 
    save_figure: bool = False, 
    output_path: str = None
) -> pd.DataFrame:
    """
    Analyzes feature importance using mutual information scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and the target column.
    parameters : dict
        Configuration parameters including:
        - "non_features_columns": List of columns to exclude from analysis.
        - "fss_use_bands": Boolean flag to include or exclude band features.
    target_col : str
        Column name for the target variable.
    save_figure : bool, optional
        If True, saves the feature importance plot (default is False).
    output_path : str, optional
        Path to save the figure (required if save_figure is True).

    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and their corresponding importance scores.
    """
    
    if save_figure and output_path == None:
        raise ValueError(f"For correlation anaysis if save figura is True an output path must be specified.\nActual values:\nsave_figure: {save_figure}\noutput_path: {output_path}")
    
    use_bands = parameters["fss_use_bands"] == True
    
    y = df[target_col]
    features = [col for col in df.columns if not col in parameters['non_features_columns']]
    
    if not use_bands:
        features = [f for f in features if not f.startswith('band_')]

    X = df[features]

    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs = -1)
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Return the figure and importance_dict
    importances_df = importances_df.set_index("feature")
    
    return importances_df

def correlation_analysis(
    df: pd.DataFrame, 
    parameters: dict, 
    save_figure: bool = False, 
    output_path: str = None
) -> pd.DataFrame:
    """
    Computes the correlation matrix for the features in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input features.
    parameters : dict
        Configuration parameters including:
        - "non_features_columns": List of columns to exclude from analysis.
        - "fss_use_bands": Boolean flag to include or exclude band features.
    save_figure : bool, optional
        If True, saves the correlation matrix plot (default is False).
    output_path : str, optional
        Path to save the figure (required if save_figure is True).

    Returns
    -------
    pd.DataFrame
        Correlation matrix of the selected features.
    """
    
    if save_figure and output_path == None:
        raise ValueError(f"For correlation anaysis if save figura is True an output path must be specified.\nActual values:\nsave_figure: {save_figure}\noutput_path: {output_path}")
    
    # Identify features
    use_bands = parameters["fss_use_bands"] == True
    features = [col for col in df.columns if not col in parameters['non_features_columns']]
    
    if not use_bands:
        features = [f for f in features if not f.startswith("band_")]
    
    # Compute correlation matrix
    corr_matrix = df[features].corr()
    
    return corr_matrix


def store_ma_output(
    corr_matrix: pd.DataFrame, 
    importances_df: pd.DataFrame, 
    parameters: dict, 
    note: str = ""
) -> None:
    """
    Saves feature importance bar plots and correlation matrix heatmaps as images.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix of the features.
    importances_df : pd.DataFrame
        DataFrame containing feature names and mutual information scores.
    parameters : dict
        Configuration parameters including:
        - "features_inspection_path": Path to save the images.
        - "always_keep_bands": Boolean flag to indicate if band features are retained.
        - "verbose": If True, displays the plots.
    note : str, optional
        Additional note to append to file names and plot titles.

    Returns
    -------
    None
    """
    
    logger = logging.getLogger("store-output")
    features_importance_path = f"{parameters['features_inspection_path']}/importances_{note}.png"
    features_corr_matrix_path = f"{parameters['features_inspection_path']}/correlation_matrix_{note}.png"
    use_bands = parameters["fss_use_bands"] == True
    features = list(importances_df.index)
    
    if parameters["verbose"] == True:
    
        logger.info(f"[{note}] Saving features correlation and importance reports:")
        logger.info(f"- Band features used ? {use_bands}")
        logger.info(f"- Saving imporances at {features_importance_path}")
        logger.info(f"- Saving correlation matrix at {features_corr_matrix_path}")
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, max(0.2 * len(importances_df), 3)))
    sns.barplot(x='importance', y='feature', data=importances_df, ax=ax)
    ax.set_title(f"[{note}] {len(importances_df)} Feature Importances{' (No Bands)' if not use_bands else ''}")
    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    
    fig.savefig(features_importance_path)

    if parameters["verbose"] == True:
        plt.show()

    plt.close(fig)
    
    # Create the plot (no show here)
    fig, ax = plt.subplots(figsize=(max(0.7 * len(features), 4), max(0.7 * len(features), 4)))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title(f"[{note}] Correlation Matrix", fontsize=16)
    plt.tight_layout()
        
    fig.savefig(features_corr_matrix_path)
    
    if parameters["verbose"] == True:
        plt.show()
    plt.close(fig)

 
def store_rfecv_output(
    report: dict, 
    parameters: dict, 
    note: str = ""
) -> None:
    """
    Saves RFECV performance plots, feature ranking bar plots and comparison of selected vs. removed features.

    Parameters
    ----------
    report : dict
        RFECV results including:
        - "info": Detailed RFECV information and fitted model.
        - "features_removed": List of features removed by RFECV.
    parameters : dict
        Configuration parameters including:
        - "features_inspection_path": Path to save the images.
        - "verbose": If True, displays the plots.
    note : str, optional
        Additional note to append to file names and plot titles.

    Returns
    -------
    None
    """
    rfecv = report["info"]["rfecv"]
    selected_features = report["info"]["selected_features"]
    removed_features = report["features_removed"]
    
    # Feature Ranking Heatmap
    feature_ranks = pd.DataFrame({
        "feature": report["info"]["features_names_in"],
        "importance": rfecv.ranking_  # Invert ranking so higher values mean more important
    })
    feature_ranks = feature_ranks.sort_values(by="importance", ascending=False)
    feature_ranks["importance"] = feature_ranks["importance"]
    
    rfecv_performance_path = f"{parameters['features_inspection_path']}/rfecv_performance_{note}.png"
    features_ranking_path = f"{parameters['features_inspection_path']}/rfecv_features_ranking_{note}.png"
    features_comparison_path = f"{parameters['features_inspection_path']}/rfecv_features_comparison_{note}.png"
    
    fig = plt.figure(figsize=(15, 9))

    # Extract data
    x_vals = range(1, len(rfecv.cv_results_["n_features"]) + 1)
    y_vals = rfecv.cv_results_["mean_test_score"]
    optimal_n_features = rfecv.n_features_

    plt.plot(x_vals, y_vals, marker="o", linestyle="--", color="b", label="Cross-Validation Score")
    plt.axvline(x=optimal_n_features, color="red", linestyle=":", linewidth=2, label=f"Optimal Features: {optimal_n_features}")

    plt.xlabel("Number of Selected Features")
    plt.ylabel("F1 Macro Score")
    plt.title(f"RFECV Cross-Validation Performance vs. Number of Features")
    plt.legend()
    plt.grid(True)
    fig.savefig(rfecv_performance_path)

    if parameters["verbose"]:
        plt.show()
    plt.close(fig)

            
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(15, max(0.2 * len(feature_ranks), 3)))
    sns.barplot(x="importance", y="feature", data=feature_ranks, ax=ax)

    ax.set_title(f"[{note}] RFECV features ranking - {len(feature_ranks)} features selected \n(the less important the feature, the higher the importance value)")
    ax.set_xlabel("RFECV Ranking")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(features_ranking_path)
    
    if parameters["verbose"] == True:
        plt.show()
    plt.close(fig)
            
    # Removed vs. Selected Features Count
    fig = plt.figure(figsize=(6, 4))
    plt.bar(["Selected", "Removed"], [len(selected_features), len(removed_features)], color=["green", "red"])
    plt.xlabel("Feature Selection")
    plt.ylabel("Number of Features")
    plt.title(f"RFECV - Comparison of Selected vs. Removed Features")
    fig.savefig(features_comparison_path)
    
    if parameters["verbose"] == True:
        plt.show()
    plt.close(fig)

def apply_pca(
    df: pd.DataFrame, 
    parameters: dict, 
    plot: bool = True
) -> tuple:
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality using a specified selection strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features.
    parameters : dict
        Configuration parameters including:
        - "non_features_columns": Columns to exclude from PCA.
        - "pca_use_bands": Boolean flag to include band features.
        - "features_inspection_path": Path to save PCA plots.
        - "pca_variance_threshold": Float between 0 and 1, minimum cumulative variance to retain (used for 'cev').
        - "pca_components_selection_strategy": 'cev', 'ec', or 'average'
        - "verbose": If True, displays plots.
    plot : bool, optional
        If True, plots cumulative explained variance and elbow curve.

    Returns
    -------
    tuple
        - final_df (pd.DataFrame): Transformed dataset with selected principal components.
        - metadata (dict): Metadata about the PCA transformation.
    """

    # Identify feature columns
    variance_threshold = parameters["pca_variance_threshold"]
    strategy = parameters.get("pca_components_selection_strategy", "cev").lower()
    use_bands = parameters["pca_use_bands"] == True
    non_features = df[parameters["non_features_columns"]]
    features_cols = [col for col in df.columns if col not in parameters['non_features_columns']]
    band_features_cols = [f for f in df.columns if f.startswith('band_')]

    if not use_bands:
        features_cols = [f for f in features_cols if not f.startswith('band_')]

    # Select features
    X = df[features_cols]

    # Fit PCA
    pca = PCA()
    _ = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # CEV cutoff
    n_components_cev = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Elbow point using KneeLocator
    knee = KneeLocator(
        range(1, len(explained_variance) + 1),
        explained_variance,
        curve="convex",
        direction="decreasing"
    )
    elbow_point = knee.knee or 1  # fallback to 1 if None

    # Average
    n_components_avg = round((n_components_cev + elbow_point) / 2)

    # Select number of components based on strategy
    if strategy == "cev":
        n_components = n_components_cev
    elif strategy == "ec":
        n_components = elbow_point
    elif strategy == "average":
        n_components = n_components_avg
    else:
        raise ValueError(f"Invalid PCA selection strategy: {strategy}. Use 'cev', 'ec', or 'average'.")

    # Plot Cumulative Explained Variance
    if plot:
        cev_chart_path = f"{parameters['features_inspection_path']}/PCA_cumulative_explained_variance.png"
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'Cumulative Explained Variance vs Number of Components (Features used: {len(features_cols)})')

        # Add vertical lines for selected components
        plt.axvline(x=n_components_cev, color='g', linestyle='--', label=f'CEV: {n_components_cev}')
        plt.axvline(x=n_components_avg, color='purple', linestyle='--', label=f'Average: {n_components_avg}')
        plt.axvline(x=n_components, color='r', linestyle='--', label=f'Selected: {n_components}')
        #plt.axhline(y=variance_threshold, color='gray', linestyle=':', label=f'{variance_threshold*100:.1f}% Threshold')

        plt.legend()
        plt.grid(True)
        plt.savefig(cev_chart_path)
        if parameters["verbose"]:
            plt.show()
        plt.close()

    # Plot Elbow Curve
    if plot:
        ec_chart_path = f"{parameters['features_inspection_path']}/PCA_elbow_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components (Elbow Curve)')

        plt.axvline(x=elbow_point, color='orange', linestyle='--', label=f'Elbow: {elbow_point}')
        plt.axvline(x=n_components_avg, color='purple', linestyle='--', label=f'Average: {n_components_avg}')
        plt.axvline(x=n_components, color='r', linestyle='--', label=f'Selected: {n_components}')

        plt.grid(True)
        plt.legend()
        plt.savefig(ec_chart_path)
        if parameters["verbose"]:
            plt.show()
        plt.close()

    # Apply PCA with selected number of components
    pca_optimal = PCA(n_components=n_components)
    X_pca_optimal = pca_optimal.fit_transform(X)

    # Rebuild DataFrame
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca_optimal, columns=pca_columns, index=df.index)

    if not use_bands:
        pca_df = pd.concat([pca_df, df[band_features_cols]], axis=1)

    final_df = pd.concat([non_features, pca_df], axis=1)

    # Collect metadata
    metadata = {
        "method": "PCA",
        "params": {
            "selection_strategy": strategy,
            "n_components_": n_components,
            "elbow_point": elbow_point,
            "n_components_cev": n_components_cev,
            "n_components_avg": n_components_avg,
            "variance_threshold": variance_threshold,
            "components_": pca_optimal.components_,
            "explained_variance_": pca_optimal.explained_variance_,
            "explained_variance_ratio_": pca_optimal.explained_variance_ratio_,
            "singular_values_": pca_optimal.singular_values_,
            "mean_": pca_optimal.mean_,
            "n_samples_": pca_optimal.n_samples_,
            "noise_variance_": pca_optimal.noise_variance_,
            "n_features_in_": pca_optimal.n_features_in_,
            "feature_names_in_": pca_optimal.feature_names_in_,
        }
    }

    return final_df, metadata
