from siapy.optimizers.parameters import ParametersDictType

CP = "categorical_parameters"
IP = "int_parameters"
FP = "float_parameters"

xgb: ParametersDictType = {
    IP: [
        {"name": "xgb__n_estimators", "low": 100, "high": 1000},
        {"name": "xgb__max_depth", "low": 3, "high": 8},
    ],
    FP: [
        {"name": "xgb__learning_rate", "low": 0.01, "high": 0.2},
        {"name": "xgb__min_child_weight", "low": 1.0, "high": 10.0},
        {"name": "xgb__subsample", "low": 0.5, "high": 0.8},
        {"name": "xgb__colsample_bytree", "low": 0.5, "high": 0.8},
        {"name": "xgb__reg_lambda", "low": 1.0, "high": 10.0},
    ],
    CP: [
        {"name": "xgb__random_state", "choices": [1]},
    ],
}

savgol: ParametersDictType = {
    CP: [
        {"name": "savgol__win_length", "choices": [3, 5, 7, 9, 11, 13, 15]},
    ],
}

savgol_xgb: ParametersDictType = {
    IP: xgb[IP],
    FP: xgb[FP],
    CP: xgb[CP] + savgol[CP],
}

PARAMETERS = {
    "savgol-xgb": savgol_xgb,
}


def import_parameters(model_name: str) -> ParametersDictType:
    if model_name not in PARAMETERS:
        raise ValueError(f"Model '{model_name}' not found in PARAMETERS")
    return PARAMETERS[model_name]
