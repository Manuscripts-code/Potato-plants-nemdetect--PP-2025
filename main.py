import json
from typing import Optional

import source.core.validation as validation
import typer
from pydantic.json import pydantic_encoder
from source.analysis import metrics, plots, present
from source.core import artifacts, logger, settings
from source.core.artifacts import DirParams
from source.dataloader import DataLoader
from source.trainer import Trainer
from source.utils import shap

app = typer.Typer()


@app.command()
def display_settings():
    logger.info(json.dumps(settings.model_dump(), default=pydantic_encoder, indent=4))


@app.command()
def test_load_data(
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = None,
    camera_label: Optional[list[str]] = None,
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    loader = DataLoader()
    loader.load_datasets(group_id, imaging_id, camera_label)


@app.command()
def train_model(
    model: str,
    do_optimize: bool = False,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y, _ = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )

    trainer = Trainer(model)

    if do_optimize:
        study = trainer.optimize(X, y)
        artifacts.save_study(study)
    else:
        score = trainer.score_model(X, y)
        artifacts.save_metric(score)
    artifacts.save_encoder(trainer.encoder)


@app.command()
def generate_metrics(
    model: str,
    do_optimize: bool = False,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y, meta = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )
    model_ = artifacts.load_unfit_model()
    encoder = artifacts.load_encoder()

    metrics_ = metrics.calculate_metrics(model_, encoder, X, y, meta)
    artifacts.save_metrics(metrics_)


@app.command()
def generate_plots(
    model: str,
    do_optimize: bool = False,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y, meta = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )
    model_ = artifacts.load_unfit_model()
    encoder = artifacts.load_encoder()

    artifacts.save_confusion_matrix_plot(
        plots.confusion_matrix_display(model_, encoder, X, y)
    )
    artifacts.save_signatures_plot(
        plots.signatures_display(encoder, X, y, settings.bands)
    )
    shap_values = artifacts.load_shap_values()
    artifacts.save_relevant_amplitudes_plot(
        plots.relevant_amplitudes(shap_values, settings.bands)
    )
    artifacts.save_relevant_features_plot(
        plots.relevant_features(shap_values, settings.bands)
    )
    artifacts.save_umap_plot(plots.umap_display(encoder, X, y, meta))


@app.command()
def calculate_relevances(
    model: str,
    do_optimize: bool = False,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y, _ = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )
    model_ = artifacts.load_unfit_model()
    encoder = artifacts.load_encoder()

    shap_values = shap.extract_values(model_, encoder, X, y)
    artifacts.save_shap_values(shap_values)


@app.command()
def run_all(
    model: str,
    do_optimize: bool = False,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = [1, 2, 3],
    camera_label: Optional[list[str]] = ["vnir", "swir"],
):
    train_model(
        model=model,
        do_optimize=do_optimize,
        group_id=group_id,
        imaging_id=imaging_id,
        camera_label=camera_label,
    )
    generate_metrics(
        model=model,
        do_optimize=do_optimize,
        group_id=group_id,
        imaging_id=imaging_id,
        camera_label=camera_label,
    )
    calculate_relevances(
        model=model,
        do_optimize=do_optimize,
        group_id=group_id,
        imaging_id=imaging_id,
        camera_label=camera_label,
    )
    generate_plots(
        model=model,
        do_optimize=do_optimize,
        group_id=group_id,
        imaging_id=imaging_id,
        camera_label=camera_label,
    )


@app.command()
def display_metrics(
    model: Optional[str] = None,
    do_optimize: Optional[bool] = None,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = None,
    camera_label: Optional[list[str]] = None,
):
    metrics_ = artifacts.load_metrics()
    if metrics_:
        present.display_metrics(
            metrics=metrics_,
            model=model,
            do_optimize=do_optimize,
            group_id=group_id,
            imaging_id=imaging_id,
            camera_label=camera_label,
        )


if __name__ == "__main__":
    app()
