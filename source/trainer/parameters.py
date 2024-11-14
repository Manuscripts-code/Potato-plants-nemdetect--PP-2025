from siapy.optimizers.parameters import ParametersDictType

CP = "categorical_parameters"
IP = "int_parameters"
FP = "float_parameters"

# Preprocessing

savgol: ParametersDictType = {
    IP: [
        {"name": "savgol__win_length", "low": 3, "high": 15, "step": 2},
    ],
}

fft: ParametersDictType = {
    FP: [
        {"name": "fft__shape_param", "low": 0.1, "high": 2, "step": 0.1},
        {"name": "fft__sigma", "low": 0.1, "high": 5, "step": 0.01},
    ],
}

# Dimensionality reduction

pls: ParametersDictType = {
    IP: [
        {"name": "pls__n_components", "low": 3, "high": 20},
    ],
}

ica: ParametersDictType = {
    IP: [
        {"name": "ica__n_components", "low": 3, "high": 20},
        {"name": "ica__max_iter", "low": 200, "high": 1000},
    ],
    FP: [
        {"name": "ica__tol", "low": 1e-4, "high": 1e-1, "log": True},
    ],
    CP: [
        {"name": "ica__algorithm", "choices": ["parallel", "deflation"]},
        {"name": "ica__fun", "choices": ["logcosh", "exp", "cube"]},
        {"name": "ica__random_state", "choices": [0]},
    ],
}

kpca: ParametersDictType = {
    IP: [
        {"name": "kpca__n_components", "low": 3, "high": 20},
    ],
    FP: [
        {"name": "kpca__gamma", "low": 1e-7, "high": 0.1, "log": True},
        {"name": "kpca__coef0", "low": 1, "high": 10000, "log": True},
    ],
    CP: [
        {"name": "kpca__kernel", "choices": ["poly", "rbf"]},
    ],
}

# Classifiers

xgb: ParametersDictType = {
    IP: [
        {"name": "xgb__n_estimators", "low": 100, "high": 1000},
        {"name": "xgb__max_depth", "low": 3, "high": 10},
    ],
    FP: [
        {"name": "xgb__learning_rate", "low": 1e-3, "high": 10, "log": True},
        {"name": "xgb__min_child_weight", "low": 1e-1, "high": 10.0, "log": True},
        {"name": "xgb__subsample", "low": 0.5, "high": 1.0},
        {"name": "xgb__colsample_bytree", "low": 0.5, "high": 1.0},
        {"name": "xgb__reg_lambda", "low": 1.0, "high": 10.0, "log": True},
        {"name": "xgb__gamma", "low": 1e-3, "high": 5.0, "log": True},
        {"name": "xgb__reg_alpha", "low": 1e-3, "high": 5.0, "log": True},
    ],
    CP: [
        {"name": "xgb__random_state", "choices": [0]},
    ],
}

svc: ParametersDictType = {
    FP: [
        {"name": "svc__C", "low": 0.1, "high": 10000, "log": True},
        {"name": "svc__gamma", "low": 1e-7, "high": 1, "log": True},
    ],
    CP: [
        {"name": "svc__random_state", "choices": [0]},
    ],
}


def _merge_parameters(*params_dicts: ParametersDictType) -> ParametersDictType:
    merged_params: ParametersDictType = {IP: [], FP: [], CP: []}
    for params in params_dicts:
        for key in [IP, FP, CP]:
            if key in params:
                merged_params[key].extend(params[key])
    return merged_params


savgol_svc: ParametersDictType = _merge_parameters(savgol, svc)
savgol_xgb: ParametersDictType = _merge_parameters(savgol, xgb)
savgol_pls_svc: ParametersDictType = _merge_parameters(savgol, pls, svc)
savgol_pls_xgb: ParametersDictType = _merge_parameters(savgol, pls, xgb)
savgol_ica_svc: ParametersDictType = _merge_parameters(savgol, ica, svc)
savgol_ica_xgb: ParametersDictType = _merge_parameters(savgol, ica, xgb)
savgol_kpca_svc: ParametersDictType = _merge_parameters(savgol, kpca, svc)
savgol_kpca_xgb: ParametersDictType = _merge_parameters(savgol, kpca, xgb)
fft_svc: ParametersDictType = _merge_parameters(fft, svc)
fft_xgb: ParametersDictType = _merge_parameters(fft, xgb)
fft_pls_svc: ParametersDictType = _merge_parameters(fft, pls, svc)
fft_pls_xgb: ParametersDictType = _merge_parameters(fft, pls, xgb)
fft_ica_svc: ParametersDictType = _merge_parameters(fft, ica, svc)
fft_ica_xgb: ParametersDictType = _merge_parameters(fft, ica, xgb)
fft_kpca_svc: ParametersDictType = _merge_parameters(fft, kpca, svc)
fft_kpca_xgb: ParametersDictType = _merge_parameters(fft, kpca, xgb)

PARAMETERS = {
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


def import_parameters(model_name: str) -> ParametersDictType:
    if model_name not in PARAMETERS:
        raise ValueError(f"Model '{model_name}' not found in PARAMETERS")
    return PARAMETERS[model_name]
