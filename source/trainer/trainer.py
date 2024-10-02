import numpy as np
from optuna import Study
from siapy.optimizers.configs import OptimizeStudyConfig, TabularOptimizerConfig
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.parameters import TrialParameters
from siapy.optimizers.scorers import Scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .models import import_model
from .parameters import import_parameters

CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)
STUDY_CONFIG = OptimizeStudyConfig(n_trials=1, n_jobs=1)
SCORING = "f1_weighted"


class Trainer:
    def __init__(self, model_name: str):
        self._optimizer: TabularOptimizer | None = None
        self._model = import_model(model_name)
        self._parameter_opt = import_parameters(model_name)

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Study:
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

    def score_model(self, X: np.ndarray, y: np.ndarray) -> float:
        scorer = Scorer.init_cross_validator_scorer(
            scoring=SCORING,
            cv=CV,
        )
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        return scorer(model=self._model, X=X, y=y_encoded)
