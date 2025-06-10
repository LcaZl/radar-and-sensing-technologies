
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from scripting import print_dataframe
from ._pipeline_utils import standardize_dict
from ._svms import SVMS

def train_svm(
    X: np.ndarray, 
    y: np.ndarray, 
    parameters_grid: dict, 
    cv: int | str, 
    verbose: bool = False
) -> GridSearchCV:
    """
    Trains an SVM model using GridSearchCV for hyperparameter tuning.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for training.
    y : np.ndarray
        Target labels for training.
    parameters_grid : dict
        Grid of hyperparameters for SVM tuning.
    cv : int or str
        Cross-validation strategy for GridSearchCV.
    verbose : bool, optional
        If True, displays detailed output during training.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object with the best estimator.
    """
    
    # Initialize the SVC model
    svc = SVC()

    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=parameters_grid,
        cv=cv,
        scoring='f1_macro',
        verbose=verbose,
        n_jobs=os.cpu_count() or -1
    )
    grid_search.fit(X, y)

    return grid_search

def train_binary_svms(
    parameters: dict, 
    parameters_grid: dict, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    X_cal: np.ndarray, 
    Y_train: np.ndarray, 
    Y_test: np.ndarray, 
    Y_cal: np.ndarray
) -> dict:
    """
    Trains binary SVMs for each class using one-vs-rest strategy with GridSearchCV for hyperparameter tuning.
    Calibrates each SVM using isotonic regression.

    Parameters
    ----------
    parameters : dict
        Configuration parameters including:
        - "GS_cv": Cross-validation strategy for GridSearchCV.
    parameters_grid : dict
        Grid of hyperparameters for SVM tuning.
    X_train, X_test, X_cal : np.ndarray
        Feature matrices for training, testing and calibration.
    Y_train, Y_test, Y_cal : np.ndarray
        Corresponding target labels.
    
    Returns
    -------
    dict
        Dictionary containing:
        - SVM models for each class.
        - Calibrated versions of each SVM.
        - Performance reports including best hyperparameters and evaluation metrics.
    """

    classes = np.unique(Y_train)
    logger = logging.getLogger("train-binary-svms")
    svms = {}
    
    for i, label in enumerate(classes):
      
        
        y_train_curr = np.where(Y_train == label, 1, 0)
        y_test_curr = np.where(Y_test == label, 1, 0)
        y_cal_curr = np.where(Y_cal == label, 1, 0)
        
        if parameters["verbose"] == True:
            logger.info(f"Current class: {label}")
            logger.info(f"Train Pos/Neg Distribution : {np.unique(y_train_curr, return_counts=True)}")
            logger.info(f"Test Pos/Neg Distribution  : {np.unique(y_test_curr, return_counts=True)}")
            logger.info(f"Cal Pos/Neg Distribution   : {np.unique(y_cal_curr, return_counts=True)}")

            logger.info(f"Training SVM ...")

        curr_grid_search = train_svm(X_train, y_train_curr, parameters_grid, cv=3)
        
        svm = curr_grid_search.best_estimator_
        best_params = curr_grid_search.best_params_
        best_params_std = standardize_dict(best_params, f"SVMS_params_{label}")
        nc_report_raw = test_svm(svm, X_test, y_test_curr, verbose=parameters["verbose"])

        if parameters["verbose"] == True:
            logger.info(f"Calibrating SVM ...")
            
        svm_calibrated = CalibratedClassifierCV(svm, method='isotonic', cv='prefit')
        svm_calibrated.fit(X_cal, y_cal_curr)
        
        c_report_raw = test_svm(svm_calibrated, X_test, y_test_curr, verbose=parameters["verbose"])
        c_report = standardize_dict(c_report_raw, f"SVMS_EVALCAL_{label}")
        
        report = standardize_dict(nc_report_raw, f"SVMS_EVAL_{label}")
        report.update(best_params_std)
        report.update(c_report)

        svms[label] = (svm, svm_calibrated, report)
      
    return svms

def test_svm(
    model: SVC,
    x_test: np.ndarray, 
    y_test: np.ndarray, 
    verbose: bool = False
) -> dict:
    """
    Evaluates an SVM model on the test set and returns a classification report.

    Parameters
    ----------
    model : SVC
        Trained SVM model to be evaluated.
    x_test : np.ndarray
        Feature matrix for testing.
    y_test : np.ndarray
        True labels for the test dataset.
    verbose : bool, optional
        If True, displays detailed output of the classification report.

    Returns
    -------
    dict
        Classification report as a dictionary, including:
        - Precision, recall, f1-score for each class.
        - Confusion matrix as a flattened array.
    """
    logger = logging.getLogger("test-svm")

    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    report["confusion_matrix"] = cm.flatten()
            
    if verbose:
        logger.info(f"    Output shapes:")
        logger.info(f"        Y prediction   : {y_pred.shape}")
        logger.info(f"        Y test names   : {len(y_test)}")
        logger.info(f"        Confusion Matr.: {cm.shape}")
        logger.info(f"    Classification Report:\n")
        print(classification_report(y_test, y_pred, zero_division=0))    

            
    return report


def SVMS_prediction_test(
    model: SVMS, 
    X_test: np.ndarray, 
    Y_test: np.ndarray, 
    n_samples: int = None,
    chart_title = ""
) -> None:
    """
    Tests the prediction performance of an SVMS model by generating predicted labels and probabilities.
    Displays a bar chart of predicted label distributions and some samples with prediction details.

    Parameters
    ----------
    model : SVMS
        Trained SVMS model for prediction.
    X_test : np.ndarray
        Feature matrix for testing.
    Y_test : np.ndarray
        True labels for the test dataset.
    n_samples : int, optional
        Number of samples to inspect in the output. None as default indicate no limits in number of inspected samples.

    Returns
    -------
    None
    """
    
    logger = logging.getLogger("SVMS-prediction-test")
    
    logger.info(f"Model prediction test:")
    
    # Predict probabilities and labels
    probabilities = model.predict_proba(X_test)
    predicted_labels = model.predict(X_test)

    # Limit to n_samples for inspection
    probabilities = probabilities[:n_samples]
    predicted_labels = predicted_labels[:n_samples]
    true_labels = Y_test[:n_samples]

    # Build the DataFrame
    class_columns = model.classes  # Class names as column headers for probabilities
    prob_df = pd.DataFrame(probabilities, columns=class_columns)
    result_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels
    })

    # Reset indices of both DataFrames to align properly
    result_df.reset_index(drop=True, inplace=True)
    prob_df.reset_index(drop=True, inplace=True)

    # Combine true labels, predicted labels, and probabilities
    final_df = pd.concat([result_df, prob_df], axis=1)
    
    # Count the occurrences of each label
    label_counts = final_df['Predicted Label'].value_counts()

    # Plot the distribution as a bar chart
    plt.figure(figsize=(8, 6))
    ax = label_counts.plot(kind='bar', edgecolor='black')
    plt.title(chart_title, fontsize=10)
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add counts on top of each bar
    for i, count in enumerate(label_counts):
        ax.text(i, count + max(label_counts) * 0.01, str(count), ha='center', va='bottom', fontsize=10)

    # Show the chart
    print_dataframe(final_df, limit=10, title="\nPrediction details for the first 30 samples")
    plt.show()


        
