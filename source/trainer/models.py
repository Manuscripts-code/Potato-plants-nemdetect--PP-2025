from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .methods import PLSRegressionWrapper
from .preprocess import SavgolWrapper

savgol_xgb = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_svc = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)


savgol_pls_xgb = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_pls_svc = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)

MODELS = {
    "savgol-xgb": savgol_xgb,
    "savgol-svc": savgol_svc,
    "savgol-pls-xgb": savgol_pls_xgb,
    "savgol-pls-svc": savgol_pls_svc,
}


def import_model(model_name: str) -> BaseEstimator:
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found in MODELS")
    return MODELS[model_name]
