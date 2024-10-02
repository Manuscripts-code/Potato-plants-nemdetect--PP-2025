import json
from typing import Optional

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
    group_id: int,
    imaging_id: Optional[list[int]] = None,
    camera_label: Optional[list[str]] = None,
):
    loader = DataLoader()
    loader.load_datasets(group_id, imaging_id, camera_label)


def init_artifacts(model: str, group_id: int, is_optimized: bool):
    #! TODO: This is temporary development hack
    imagings_ids = [1, 2, 3]
    cameras_labels = ["vnir", "swir"]

    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=is_optimized,
            load_group_id=group_id,
            load_imagings_ids=imagings_ids,
            load_cameras_labels=cameras_labels,
        )
    )
    return imagings_ids, cameras_labels


@app.command()
def score_model(model: str, group_id: int):
    imagings_ids, cameras_labels = init_artifacts(model, group_id, False)
    X, y = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imagings_ids,
        cameras_labels=cameras_labels,
    )
    trainer = Trainer(model)
    score = trainer.score_model(X, y)
    artifacts.save_metric(score)


@app.command()
def optimize_model(model: str, group_id: int):
    imagings_ids, cameras_labels = init_artifacts(model, group_id, True)
    X, y = DataLoader().load_datasets(
        group_id=group_id,
        imagings_ids=imagings_ids,
        cameras_labels=cameras_labels,
    )
    trainer = Trainer(model)
    study = trainer.optimize(X, y)
    artifacts.save_study(study)


if __name__ == "__main__":
    app()
