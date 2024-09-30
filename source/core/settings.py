from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = ".env"

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SAVED_DIR = BASE_DIR / "outputs"

SAVED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True, dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    data_dir: Path = Field(
        default=BASE_DIR / "data",
        description="Path to data directory.",
    )
    outputs_dir: Path = Field(
        default=SAVED_DIR,
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

    file_labels: str = "labels.csv"
    file_signatures_vnir: str = "signatures_vnir.csv"
    file_signatures_swir: str = "signatures_swir.csv"

    dataset_id_map: dict = {1: "imaging1", 2: "imaging2", 3: "imaging3"}
    cameras_id_labels: list = ["vnir", "swir"]


settings = Settings()
