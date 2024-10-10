from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from source.utils.utils import read_json

DOTENV_PATH = ".env"
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
load_dotenv(override=True, dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    data_dir: Path = Field(
        default=BASE_DIR / "data",
        description="Path to data directory.",
    )
    outputs_dir: Path = Field(
        default=BASE_DIR / "outputs",
        description="Path to program outputs directory.",
    )
    debug: bool = Field(
        default=False, description="If logging displays debug information."
    )

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        cli_parse_args=False,
        env_ignore_empty=True,
    )

    @property
    def bands(self) -> np.ndarray:
        return np.array(read_json(self.data_dir / "bands.json")["SPECTRAL_BANDS"])  # type: ignore


settings = Settings()
