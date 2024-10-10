from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from optuna import Study
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from source.analysis import present
from source.analysis.metrics import Metrics
from source.trainer.models import import_model
from source.utils.specific import DirParams, params_to_path
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
RESULTS = "results"
RESULTS_METRICS = "metrics.txt"
RESULT_SHAP_VALUES = "shap_values.npy"
RESULTS_UMAP = "umap.png"
RESULT_CONFUSION_MTX = "confusion_matrix.png"
RESULT_RELEVANT_FEATURES = "relevant_features.png"
RESULT_RELEVANT_AMPLITUDES = "relevant_amplitudes.png"
RESULT_SIGNATURES = "signatures.png"


class Artifacts:
    def __init__(self):
        self._dir_params: DirParams
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
        self._dir_params = params
        self._artifacts_path = OUT_DIR / params_to_path(params)

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

    def load_encoder(self) -> LabelEncoder:
        save_path = self._get_save_path(STUDY, STUDY_ENCODER)
        if save_path:
            return read_pickle(save_path)
        raise ValueError(
            "Encoder could not be found. Make sure you train the model first (cmd: train_model)"
        )

    def load_unfit_model(self) -> BaseEstimator:
        params = self.load_params()
        model = import_model(self._dir_params.estimator_name)
        if params:
            model.set_params(**params)
        return model

    def save_metrics(self, metrics: list[Metrics]):
        table, metrics_all = present.generate_metrics_table(metrics)
        save_path = self._set_save_path(RESULTS)
        write_txt(table, save_path / RESULTS_METRICS)
        _ = [write_txt(f"{m.mean:.2f}", save_path / f"{m.name}") for m in metrics_all]

    def load_metrics(self) -> Optional[dict[str, list[Metrics]]]:
        if not OUT_DIR:
            logger.warning("No output directory.")
            return None

        metrics_all = {}
        for metric_all_file in sorted(OUT_DIR.glob("*")):
            metrics_files = metric_all_file / RESULTS
            metrics = []
            for metric_file in sorted(metrics_files.glob("*")):
                if not metric_file.suffix:
                    mean_value = float(metric_file.read_text())
                    name = metric_file.stem
                    metrics.append(Metrics(name=name, mean=mean_value))
            if metrics:
                metrics_all[metric_all_file.stem] = metrics

        return metrics_all

    def save_plot(self, diagram: Figure, filename: str):
        plt.style.use("default")
        save_path = self._set_save_path(RESULTS)
        diagram.savefig(save_path / filename, format="png", bbox_inches="tight")
        plt.close(diagram)

    def save_umap_plot(self, diagram: Figure):
        self.save_plot(diagram, RESULTS_UMAP)

    def save_confusion_matrix_plot(self, diagram: Figure):
        self.save_plot(diagram, RESULT_CONFUSION_MTX)

    def save_relevant_features_plot(self, diagram: Figure):
        self.save_plot(diagram, RESULT_RELEVANT_FEATURES)

    def save_relevant_amplitudes_plot(self, diagram: Figure):
        self.save_plot(diagram, RESULT_RELEVANT_AMPLITUDES)

    def save_signatures_plot(self, diagram: Figure):
        self.save_plot(diagram, RESULT_SIGNATURES)

    def save_shap_values(self, values: np.ndarray):
        save_path = self._set_save_path(RESULTS)
        np.save(save_path / RESULT_SHAP_VALUES, values)

    def load_shap_values(self) -> np.ndarray:
        save_path = self._get_save_path(RESULTS, RESULT_SHAP_VALUES)
        if save_path:
            return np.load(save_path)
        raise ValueError(
            "SHAP values file could not be found. Make sure you have saved the SHAP values."
        )


artifacts = Artifacts()
