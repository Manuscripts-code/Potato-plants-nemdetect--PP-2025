from pydantic import BaseModel

from .settings import settings

OUT_DIR = settings.outputs_dir


class DirParams(BaseModel):
    name_model: str
    load_group_id: int
    load_imagings_ids: list[int]
    load_cameras_labels: list[str]


class Artifacts:
    def __init__(self):
        self._dir_params: DirParams

    def set_dir_params(self, dir_params: DirParams):
        self._dir_params = dir_params


artifacts = Artifacts()
