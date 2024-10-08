from sklearn.base import BaseEstimator
from sklearn.decomposition import FastICA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .methods import PLSRegressionWrapper
from .preprocess import DetrendTransformer, SavgolWrapper, SNVTransformer  # noqa: F401

savgol_xgb = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_svc = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)


savgol_pls_xgb = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_pls_svc = Pipeline(
    [
        # ("detrend", DetrendTransformer(type="constant")),
        # ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)

savgol_ica_svc = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("ica", FastICA(random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)

savgol_ica_xgb = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("ica", FastICA(random_state=0)),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_kpca_svc = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("kpca", KernelPCA(kernel="rbf", random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)
savgol_kpca_xgb = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("kpca", KernelPCA(kernel="rbf", random_state=0)),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

MODELS = {
    "savgol-xgb": savgol_xgb,
    "savgol-svc": savgol_svc,
    "savgol-pls-xgb": savgol_pls_xgb,
    "savgol-pls-svc": savgol_pls_svc,
    "savgol-ica-svc": savgol_ica_svc,
    "savgol-ica-xgb": savgol_ica_xgb,
    "savgol-kpca-svc": savgol_kpca_svc,
    "savgol-kpca-xgb": savgol_kpca_xgb,
}


def import_model(model_name: str) -> BaseEstimator:
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found in MODELS")
    return MODELS[model_name]
