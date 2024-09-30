import json
from typing import Literal

import typer
from pydantic.json import pydantic_encoder
from source.core import logger, settings

app = typer.Typer()


@app.command()
def display_settings():
    logger.info(json.dumps(settings.model_dump(), default=pydantic_encoder, indent=4))


@app.command()
def test_load_data(dataset: int):
    pass


if __name__ == "__main__":
    app()
