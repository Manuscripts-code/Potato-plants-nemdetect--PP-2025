from sklearn.base import BaseEstimator
from sklearn.decomposition import FastICA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .methods import PLSRegressionWrapper
from .preprocess import FFTWrapper, SavgolWrapper, SNVTransformer

savgol_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)


savgol_pls_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_pls_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)

savgol_ica_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("ica", FastICA(random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)

savgol_ica_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("ica", FastICA(random_state=0)),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

savgol_kpca_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("kpca", KernelPCA(kernel="rbf", random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)
savgol_kpca_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("savgol", SavgolWrapper()),
        ("kpca", KernelPCA(kernel="rbf", random_state=0)),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

fft_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

fft_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)

fft_pls_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

fft_pls_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("pls", PLSRegressionWrapper()),
        ("svc", SVC(random_state=0)),
    ]
)

fft_ica_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("ica", FastICA(random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)

fft_ica_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("ica", FastICA(random_state=0)),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

fft_kpca_svc = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
        ("kpca", KernelPCA(kernel="rbf", random_state=0)),
        ("svc", SVC(random_state=0)),
    ]
)

fft_kpca_xgb = Pipeline(
    [
        ("snv", SNVTransformer()),
        ("scaler", StandardScaler()),
        ("fft", FFTWrapper()),
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
    "fft-xgb": fft_xgb,
    "fft-svc": fft_svc,
    "fft-pls-xgb": fft_pls_xgb,
    "fft-pls-svc": fft_pls_svc,
    "fft-ica-svc": fft_ica_svc,
    "fft-ica-xgb": fft_ica_xgb,
    "fft-kpca-svc": fft_kpca_svc,
    "fft-kpca-xgb": fft_kpca_xgb,
}


def import_model(model_name: str) -> BaseEstimator:
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found in MODELS")
    return MODELS[model_name]
