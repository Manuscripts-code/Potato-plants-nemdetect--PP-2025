from pathlib import Path

from pydantic import BaseModel


class DirParams(BaseModel):
    estimator_name: str
    estimator_is_optimized: bool
    load_group_id: int
    load_imagings_ids: list[int]
    load_cameras_labels: list[str]


def params_to_path(params: DirParams) -> str:
    return "__".join(
        [
            params.estimator_name,
            str(params.load_group_id),
            "-".join(params.load_cameras_labels),
            "-".join([str(ii) for ii in params.load_imagings_ids]),
            str(params.estimator_is_optimized),
        ]
    )


def params_from_path(path: str | Path) -> DirParams:
    path = Path(path)
    path = path.stem
    parts = path.split("__")

    estimator_name = parts[0]
    load_group_id = int(parts[1])
    load_cameras_labels = parts[2].split("-")
    load_imagings_ids = [int(ii) for ii in parts[3].split("-")]
    estimator_is_optimized = parts[4].lower() == "true"

    return DirParams(
        estimator_name=estimator_name,
        load_group_id=load_group_id,
        load_cameras_labels=load_cameras_labels,
        load_imagings_ids=load_imagings_ids,
        estimator_is_optimized=estimator_is_optimized,
    )
