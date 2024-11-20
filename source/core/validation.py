from typing import Optional


def check_group_id(group_id: Optional[int]) -> int:
    if group_id is None:
        raise ValueError("Group ID must be specified. \n Usage: --group-id")
    return group_id


def check_imaging_id(imaging_id: Optional[list[int]]) -> list[int]:
    if imaging_id is None or not imaging_id:
        raise ValueError(
            "At least one Imaging ID must be specified and cannot be empty. \n Usage: --imaging-id"
        )
    return imaging_id


def check_camera_label(camera_label: Optional[list[str]]) -> list[str]:
    if camera_label is None or not camera_label:
        raise ValueError(
            "At least one Camera label must be specified and cannot be empty. \n Usage: --camera-label"
        )
    return camera_label
