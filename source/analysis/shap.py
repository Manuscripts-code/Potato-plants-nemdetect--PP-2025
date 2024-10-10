import numpy as np
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder


def extract_values(
    model: BaseEstimator,
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    model = clone(model)
    y_encoded = np.array(encoder.fit_transform(y))
    model.fit(X, y_encoded)  # type: ignore
    explainer = shap.KernelExplainer(model.predict, X)  # type: ignore
    shap_values = explainer.shap_values(X)
    return shap_values
