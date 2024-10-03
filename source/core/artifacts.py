from pathlib import Path
from typing import Optional

from optuna import Study
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from source.trainer.models import import_model
from source.utils.utils import (
    read_json,
    read_pickle,
    write_json,
    write_pickle,
    write_txt,
)

from .logger import logger
from .settings import settings

OUT_DIR = settings.outputs_dir
STUDY = "study"
STUDY_BEST_PARAMS = "best_params.json"
STUDY_BEST_METRIC = "best_metric.txt"
STUDY_ENCODER = "encoder.pkl"


class DirParams(BaseModel):
    estimator_name: str
    estimator_is_optimized: bool
    load_group_id: int
    load_imagings_ids: list[int]
    load_cameras_labels: list[str]


class Artifacts:
    def __init__(self):
        self._artifacts_path: Path

    def _set_save_path(self, dir_name: str) -> Path:
        save_path = self._artifacts_path / dir_name
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _get_save_path(self, dir_name: str, file_name: str) -> Optional[Path]:
        save_path = self._artifacts_path / dir_name / file_name
        if save_path.exists():
            return save_path
        return None

    def set_dir_params(self, params: DirParams):
        self._artifacts_path = OUT_DIR / "__".join(
            [
                params.estimator_name,
                str(params.load_group_id),
                "-".join(params.load_cameras_labels),
                "-".join([str(ii) for ii in params.load_imagings_ids]),
                str(params.estimator_is_optimized),
            ]
        )

    def save_study(self, study: Study):
        self.save_metric(study.best_value)
        self.save_params(study.best_params)

    def save_metric(self, metric: float):
        save_path = self._set_save_path(STUDY)
        write_txt(str(metric), save_path / STUDY_BEST_METRIC)
        logger.info(f"Metric saved: {metric}")

    def save_params(self, params: dict):
        save_path = self._set_save_path(STUDY)
        write_json(params, save_path / STUDY_BEST_PARAMS)
        logger.info(f"Params saved: {params}")

    def load_params(self) -> Optional[dict]:
        save_path = self._get_save_path(STUDY, STUDY_BEST_PARAMS)
        if save_path:
            return read_json(save_path)
        return None

    def save_encoder(self, encoder: LabelEncoder):
        save_path = self._set_save_path(STUDY)
        write_pickle(encoder, save_path / STUDY_ENCODER)

    def load_encoder(self) -> Optional[LabelEncoder]:
        save_path = self._get_save_path(STUDY, STUDY_ENCODER)
        if save_path:
            return read_pickle(save_path)
        return None

    def load_unfit_model(self, model_name: str) -> BaseEstimator:
        params = self.load_params()
        model = import_model(model_name)
        if params:
            model.set_params(**params)
        return model


artifacts = Artifacts()
