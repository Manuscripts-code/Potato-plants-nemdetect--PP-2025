import pandas as pd
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.scorers import Scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .models import MODELS


class Trainer:
    def __init__(self, model: str):
        self._optimizer: TabularOptimizer | None = None
        if model in MODELS:
            self._model = MODELS[model]
        else:
            raise ValueError(f"Model '{model}' not found in MODELS")

    # def run(self, X: pd.DataFrame, y: pd.Series):
    #     self._optimizer = TabularOptimizer(
    #         model=self._model,
    #         configs=TabularOptimizerConfig(),
    #         X=X,
    #         y=y,
    #     )
    #     study = self._optimizer.run()
    #     return study

    def score_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        scorer = Scorer.init_cross_validator_scorer(
            scoring="f1",
            cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0),
        )
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        return scorer(model=self._model, X=X.to_numpy(), y=y_encoded)
