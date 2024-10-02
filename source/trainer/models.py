from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocess import SavgolWrapper

savgol_xgb = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

MODELS = {
    "savgol-xgb": savgol_xgb,
}


def import_model(model_name: str) -> BaseEstimator:
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found in MODELS")
    return MODELS[model_name]
