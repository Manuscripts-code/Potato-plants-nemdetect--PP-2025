import json
from typing import Optional

import source.core.validation as validation
import typer
from pydantic.json import pydantic_encoder
from source.core import artifacts, logger, settings
from source.core.artifacts import DirParams
from source.dataloader import DataLoader
from source.trainer import Trainer

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
def score_model(
    model: str,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = None,
    camera_label: Optional[list[str]] = None,
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=False,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )
    trainer = Trainer(model)
    score = trainer.score_model(X, y)
    artifacts.save_metric(score)


@app.command()
def optimize_model(
    model: str,
    group_id: Optional[int] = None,
    imaging_id: Optional[list[int]] = None,
    camera_label: Optional[list[str]] = None,
):
    group_id = validation.check_group_id(group_id)
    imaging_id = validation.check_imaging_id(imaging_id)
    camera_label = validation.check_camera_label(camera_label)

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=True,
            load_group_id=group_id,
            load_imagings_ids=imaging_id,
            load_cameras_labels=camera_label,
        )
    )

    X, y = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imaging_id,
        cameras_labels=camera_label,
    )
    trainer = Trainer(model)
    study = trainer.optimize(X, y)
    artifacts.save_study(study)


if __name__ == "__main__":
    app()
