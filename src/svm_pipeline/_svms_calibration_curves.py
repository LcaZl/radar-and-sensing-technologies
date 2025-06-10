import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay, calibration_curve
import numpy as np
from sklearn.preprocessing import label_binarize

def binary_svms_calibration_curves(
    model, 
    X_cal: np.ndarray, 
    Y_cal: np.ndarray, 
    output_dir: str = None, 
    show_plots: bool = True
) -> None:
    """
    Plots separeted calibration curves in a plot, one for each class and thus for each binary SVM (calibrated and uncalibrated if available).

    Parameters
    ----------
    model : object
        The trained model containing binary SVMs and their calibrated versions.
    X_cal : np.ndarray
        Feature data for calibration.
    Y_cal : np.ndarray
        True labels for the calibration dataset.
    output_dir : str, optional
        Directory to save the plots.
    show_plots : bool, optional
        If True, displays the plots.

    Returns
    -------
    None
    """

    # Prepare the plots
    fig_cal, ax_cal = plt.subplots(figsize=(16, 12))

    ax_uncal = None
    if hasattr(list(model.binary_svms.values())[0], "predict_proba"):
        fig_uncal, ax_uncal = plt.subplots(figsize=(16, 12))

    colors = plt.cm.tab10.colors 
    
    for i, label in enumerate(model.classes):
        svm = model.binary_svms[label]
        svm_cal = model.binary_svms_calibrated[label]
        
        y_cal_curr = np.where(Y_cal == label, 1, 0)

        if hasattr(svm, "predict_proba"):
            # Add uncalibrated SVM curve
            CalibrationDisplay.from_estimator(
                svm,
                X_cal,
                y_cal_curr,
                n_bins=10,
                name=label,
                ax=ax_uncal,
                color=colors[i % len(colors)],
                strategy='quantile'
            )

        # Add calibrated SVM curve
        CalibrationDisplay.from_estimator(
            svm_cal,
            X_cal,
            y_cal_curr,
            n_bins=10,
            name=label,
            ax=ax_cal,
            color=colors[i % len(colors)],
            strategy='quantile'
        )

    if ax_uncal is not None:
        ax_uncal.set_title("Uncalibrated Binary SVMs Calibration Curves")
        ax_uncal.legend(title="Class Label", loc="best")
        ax_uncal.set_xlabel("Mean Predicted Probability")
        ax_uncal.set_ylabel("Fraction of Positives")
        if output_dir:
            fig_uncal.savefig(f"{output_dir}/uncalibrated_svms_calibration_curves.png")
        if show_plots:
            plt.show()
        plt.close(fig_uncal)

    ax_cal.set_title("Calibrated Binary SVMs Calibration Curves")
    ax_cal.legend(title="Class Label", loc="best")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    if output_dir:
        fig_cal.savefig(f"{output_dir}/calibrated_svms_calibration_curves.png")
    if show_plots:
        plt.show()
    plt.close(fig_cal)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import label_binarize

def multiclass_svm_calibration_curves(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    output_dir: str = None, 
    show_plots: bool = True
) -> None:
    """
    Plots calibration curves for each class of a multi-class SVM (calibrated and uncalibrated).
    
    Parameters
    ----------
    model : object
        The trained model containing the multi-class SVM and its calibrated version.
    X_test : np.ndarray
        Feature data for testing.
    y_test : np.ndarray
        True labels for the test dataset.
    output_dir : str, optional
        Directory to save the plots.
    show_plots : bool, optional
        If True, displays the plots.

    Returns
    -------
    None
    """

    svm = model.multiclass_svm
    svm_cal = model.multiclass_svm_calibrated
    
    # -- Uncalibrated Plot --
    if hasattr(svm, "predict_proba"):
        y_prob_uncal = svm.predict_proba(X_test)

        # Binarize labels
        classes = np.unique(y_test)
        y_test_binarized = label_binarize(y_test, classes=classes)

        fig_uncal, ax_uncal = plt.subplots(figsize=(16, 12))
        for i, class_label in enumerate(classes):
            CalibrationDisplay.from_predictions(
                y_test_binarized[:, i],
                y_prob_uncal[:, i],
                n_bins=10,
                name=f"Class {class_label}",
                ax=ax_uncal,
                strategy='quantile'
            )

        # Add a diagonal "perfect calibration" line:
        ax_uncal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax_uncal.set_title("Uncalibrated SVM - Calibration Curves")
        ax_uncal.set_xlabel("Mean Predicted Probability")
        ax_uncal.set_ylabel("Fraction of Positives")
        ax_uncal.legend(loc="best")
        
        if output_dir:
            fig_uncal.savefig(f"{output_dir}/uncalibrated_svm_calibration_curves.png")
        if show_plots:
            plt.show()
        plt.close(fig_uncal)

    # -- Calibrated Plot --
    # We assume svm_cal has `predict_proba`
    y_prob_cal = svm_cal.predict_proba(X_test)

    # Binarize labels
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)

    fig_cal, ax_cal = plt.subplots(figsize=(16, 12))
    for i, class_label in enumerate(classes):
        CalibrationDisplay.from_predictions(
            y_test_binarized[:, i],
            y_prob_cal[:, i],
            n_bins=10,
            name=f"Class {class_label}",
            ax=ax_cal,
            strategy='quantile'
        )

    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax_cal.set_title("Calibrated SVM - Calibration Curves")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.legend(loc="best")

    if output_dir:
        fig_cal.savefig(f"{output_dir}/calibrated_svm_calibration_curves.png")
    if show_plots:
        plt.show()
    plt.close(fig_cal)
