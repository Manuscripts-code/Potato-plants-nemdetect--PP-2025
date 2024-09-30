import json
from typing import Optional

import typer
from pydantic.json import pydantic_encoder
from source.core import logger, settings
from source.dataloader import DataLoader

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


if __name__ == "__main__":
    app()
