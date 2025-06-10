from ._pipeline_utils import (
    standardize_dict,
    save_experiment_result,
    create_next_experiment_folder,
    reload_and_apply_PCA
)

from ._svms_calibration_curves import (
    multiclass_svm_calibration_curves,
    binary_svms_calibration_curves
)

from ._features_selection import (
    features_selection,
    rfecv_analysis,
    multicollinearity_analysis,
    store_rfecv_output,
    store_ma_output,
    correlation_analysis,
    features_importance_analysis,
    apply_pca
)

from ._features_preprocessing import (
    preprocess_features,
    split_dataset,
    get_scaler,
    DegreeToSinCos
)

from ._svms_evaluation import (
    train_svm,
    train_binary_svms,
    test_svm,
    SVMS_prediction_test
)

from ._svms import SVMS, save_SVMS, load_SVMS
from ._svms_grid_search import grid_search
#from ...doc._svms_grid_search import grid_search_SVMs, grid_search_SVM

__all__ = [
    
    "standardize_dict",
    "save_experiment_result",
    "multiclass_svm_calibration_curves",
    "binary_svms_calibration_curves",
    "create_next_experiment_folder",
    "features_selection",
    "correlation_analysis",
    "features_importance_analysis",
    "apply_pca",
    "rfecv_analysis",
    "multicollinearity_analysis",
    "store_rfecv_output",
    "store_ma_output",
    "preprocess_features",
    "preprocess_features_simple",
    "split_dataset",
    "get_scaler",
    "DegreeToSinCos",
    "train_svm",
    "train_binary_svms",
    "test_svm",
    "SVMS_prediction_test",
    #"grid_search_SVMs",
    #"grid_search_SVM",
    "SVMS",
    "save_SVMS", 
    "load_SVMS",
    "grid_search", 
    "generate_key"
]