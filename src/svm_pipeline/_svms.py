
import logging
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from scipy.special import softmax

class SVMS:
    """
    A combined SVM model class supporting both multiclass and binary one-vs-rest approaches.
    It includes calibrated and uncalibrated versions and supports softmax-based probability normalization.

    Attributes
    ----------
    verbose : bool
        If True, enables detailed logging.
    logger : logging.Logger
        Logger for printing model info.
    experiment_parameters : dict
        Experiment configuration parameters.
    classes : list
        List of class labels.
    n_classes : int
        Number of unique classes.
    id2class : dict
        Mapping from class ID to class label.
    class2id : dict
        Mapping from class label to class ID.
    multiclass_svm : SVC
        Multiclass SVM model.
    multiclass_svm_calibrated : CalibratedClassifierCV
        Calibrated version of the multiclass SVM.
    binary_svms : dict
        Dictionary of binary SVM models for one-vs-rest classification.
    binary_svms_calibrated : dict
        Dictionary of calibrated binary SVMs.
    use_multiclass_svm_calibrated : bool
        If True, uses the calibrated multiclass svm for predictions.
    use_binary_svms : bool
        If True, enables the use of binary one-vs-rest SVMs.
    use_binary_svms_with_softmax : bool
        If True, applies softmax normalization to probabilities.
    binary_svms_reports : dict
        Dictionary of performance reports for each binary SVM.
    preprocessing_metadata : dict
        Metadata about feature preprocessing steps.
    pca_metadata : dict or None
        Metadata for PCA transformation, if used
    features_used : set()
        Features of the original dataset used
    

    Methods
    -------
    predict_proba(X)
        Predicts class probabilities for the input features using Multiclass SVM, calibrated or not.
    predict(X)
        Predicts class labels for the input features.
    predict_proba_with_binary_svms_softmax(X)
        Predicts class probabilities using one-vs-rest binary SVMs with softmax normalization.
    predict_proba_with_binary_svms(X)
        Predicts class probabilities using one-vs-rest binary SVMs without softmax normalization.
    """

    def __init__(self, svm, svm_calibrated, svms, parameters, classes, preprocessing_metadata, training_report, pca = None, features_used = [], features_selected = []):

        self.verbose = False
        self.logger = logging.getLogger("SVMS")
        
        self.training_report = training_report
        self.experiment_parameters = parameters
        self.features_used = features_used
        self.features_selected = features_selected
        
        self.classes = classes
        self.n_classes = len(self.classes)

        self.id2class = {id : label for id, label in enumerate(self.classes)}
        self.class2id = {label : id for id, label in self.id2class.items()}

        self.multiclass_svm = svm
        self.multiclass_svm_calibrated = svm_calibrated
        
        self.preprocessing_metadata = preprocessing_metadata
        self.pca = pca
        
        self.binary_svms = {}
        self.binary_svms_calibrated = {}
        self.binary_svms_reports = {}
        
        for label in self.classes:
            
            svm, svm_cal, report = svms[label]
            
            self.binary_svms[label] = svm
            self.binary_svms_calibrated[label] = svm_cal
            self.binary_svms_reports[label] = report

        self.use_binary_svms = False
        self.use_binary_svms_with_softmax = False
        self.use_multiclass_svm_calibrated = False
        
    def predict_proba(self, X):
        
        if self.use_binary_svms_with_softmax and not self.use_binary_svms:
            raise ValueError("Set use_binary_svms to True to use the variant with_softmax.")
        
        if self.use_binary_svms:
            
            if self.use_binary_svms_with_softmax:
                return self.predict_proba_with_binary_svms_softmax(X)
            else:
                return self.predict_proba_with_binary_svms(X)
        
        if self.use_multiclass_svm_calibrated:
            return self.multiclass_svm_calibrated.predict_proba(X)
        else:
            return self.multiclass_svm.predict_proba(X)
        
    def predict(self, X):
        
        if self.use_binary_svms:
            proba = self.predict_proba(X)
            predicted_indices = np.argmax(proba, axis=1)
            return np.array([self.id2class[int(idx)] for idx in predicted_indices])

        if self.use_multiclass_svm_calibrated:
            return self.multiclass_svm_calibrated.predict(X)
        else:
            return self.multiclass_svm.predict(X)


    def predict_proba_with_binary_svms_softmax(self, X):
        
        proba = np.zeros((X.shape[0], self.n_classes))
        
        #decision_function = self.estimator.predict_proba(X)
        #decision_function = self.estimator.decision_function(X)
        
        if self.verbose:
            self.logger("\nPredict probabilities method inspection:")
            
        for i, cls in enumerate(self.classes):
            
            svm = self.binary_svms[cls]
            svm_cal = self.binary_svms_calibrated[cls]
            
            # Method 1
            svm_cal = svm_cal.calibrated_classifiers_[0].calibrators[0] # 1
            decision_function = svm.decision_function(X)
            proba[:, i] = svm_cal.predict(decision_function) # 1

            # Method 2
            #proba[:, i] = svm_cal.predict_proba(X)[:, 1] # 2
            
            if self.verbose:
                self.logger(f"\nRetrieving probabilities for {cls}:")
                self.logger(f"Input X shape: {X.shape}")
                self.logger(f"Decision function shape before reshape: {svm.decision_function(X).shape}")
                self.logger(f"Decision function shape after  reshape: {decision_function.shape}")
                self.logger(f"Decision function ndim : {decision_function.ndim}")
                self.logger(f"Probabilities shape    : {proba.shape}")
                
        proba = softmax(proba, axis=1)
        return proba


    def predict_proba_with_binary_svms(self, X):
        
        proba = np.zeros((X.shape[0], self.n_classes))

        if self.verbose:
            self.logger("\nPredict probabilities method inspection:")
            
        for i, cls in enumerate(self.classes):
            
            svm = self.binary_svms[cls]
            svm_cal = self.binary_svms_calibrated[cls]
            
            # Method 1
            svm_cal = svm_cal.calibrated_classifiers_[0].calibrators[0] # 1
            decision_function = svm.decision_function(X)
            proba[:, i] = svm_cal.predict(decision_function) # 1

            
            if self.verbose:
                self.logger(f"\nRetrieving probabilities for {cls}:")
                self.logger(f"Input X shape: {X.shape}")
                self.logger(f"Decision function shape before reshape: {svm.decision_function(X).shape}")
                self.logger(f"Decision function shape after  reshape: {decision_function.shape}")
                self.logger(f"Decision function ndim : {decision_function.ndim}")
                self.logger(f"Probabilities shape    : {proba.shape}")
                
        # Normalize probabilities
        if self.n_classes == 2:
            # For binary classification, ensure probabilities sum to 1
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            denominator = np.sum(proba, axis=1)[:, np.newaxis]
            
            # In the edge case where for each class calibrator returns a null
            # probability for a given sample, use the uniform distribution
            # instead.
            uniform_proba = np.full_like(proba, 1 / self.n_classes)
            proba = np.divide(
                proba, denominator, out=uniform_proba, where=denominator != 0
            )

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba
        

def save_SVMS(
    model: SVMS, 
    filepath: str
) -> None:
    """
    Saves the SVMS class instance to a file using joblib.

    Parameters
    ----------
    model : SVMS
        The SVMS model to save.
    filepath : str
        The file path to save the model.

    Returns
    -------
    None
    """
    joblib.dump(model, filepath)
    
def load_SVMS(
    filepath: str
) -> SVMS:
    """
    Loads an SVMS instance from a file.

    Parameters
    ----------
    filepath : str
        The file path to load the model from.

    Returns
    -------
    SVMS
        The loaded SVMS model.
    """
    return joblib.load(filepath)
    
