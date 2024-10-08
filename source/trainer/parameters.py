from siapy.optimizers.parameters import ParametersDictType

CP = "categorical_parameters"
IP = "int_parameters"
FP = "float_parameters"

# Peprocessing

savgol: ParametersDictType = {
    IP: [
        {"name": "savgol__win_length", "low": 3, "high": 15, "step": 2},
    ],
}

# Dimensionality reduction

pls: ParametersDictType = {
    IP: [
        {"name": "pls__n_components", "low": 3, "high": 10, "log": True},
    ],
}

# Classifiers

xgb: ParametersDictType = {
    IP: [
        {"name": "xgb__n_estimators", "low": 100, "high": 1000, "log": True},
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


savgol_xgb: ParametersDictType = _merge_parameters(savgol, xgb)
savgol_svc: ParametersDictType = _merge_parameters(savgol, svc)
savgol_pls_xgb: ParametersDictType = _merge_parameters(savgol, pls, xgb)
savgol_pls_svc: ParametersDictType = _merge_parameters(savgol, pls, svc)

PARAMETERS = {
    "savgol-xgb": savgol_xgb,
    "savgol-svc": savgol_svc,
    "savgol-pls-xgb": savgol_pls_xgb,
    "savgol-pls-svc": savgol_pls_svc,
}


def import_parameters(model_name: str) -> ParametersDictType:
    if model_name not in PARAMETERS:
        raise ValueError(f"Model '{model_name}' not found in PARAMETERS")
    return PARAMETERS[model_name]
