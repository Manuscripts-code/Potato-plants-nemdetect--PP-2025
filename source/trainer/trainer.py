import pandas as pd
from siapy.optimizers.configs import OptimizeStudyConfig, TabularOptimizerConfig
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.parameters import ParametersDictType, TrialParameters
from siapy.optimizers.scorers import Scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .models import MODELS
from .parameters import PARAMETERS

CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)
STUDY_CONFIG = OptimizeStudyConfig(n_trials=100, n_jobs=10)
SCORING = "f1_weighted"


class Trainer:
    def __init__(self, model_name: str):
        self._optimizer: TabularOptimizer | None = None
        self._model_name = model_name
        self._model: Pipeline
        self._parameter_opt: ParametersDictType
        self._check_model_existence()

    def _check_model_existence(self):
        if self._model_name not in MODELS:
            raise ValueError(f"Model '{self._model_name}' not found in MODELS")
        if self._model_name not in PARAMETERS:
            raise ValueError(f"Model '{self._model_name}' not found in PARAMETERS")
        self._model = MODELS[self._model_name]
        self._parameter_opt = PARAMETERS[self._model_name]

    def optimize(self, X: pd.DataFrame, y: pd.Series):
        trial_parameters = TrialParameters.from_dict(self._parameter_opt)
        scorer = Scorer.init_cross_validator_scorer(
            scoring=SCORING,
            cv=CV,
        )
        configs = TabularOptimizerConfig(
            trial_parameters=trial_parameters,
            scorer=scorer,
            optimize_study=STUDY_CONFIG,
        )
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        self._optimizer = TabularOptimizer(
            model=self._model,
            configs=configs,
            X=X,
            y=y_encoded,
        )
        study = self._optimizer.run()
        return study

    def score_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        scorer = Scorer.init_cross_validator_scorer(
            scoring=SCORING,
            cv=CV,
        )
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        return scorer(model=self._model, X=X, y=y_encoded)
