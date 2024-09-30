from typing import Optional

import pandas as pd

from source.core import settings

from .helpers import get_labels_by_group


def get_coresponding_signatures(signatures_list, labels_list, items_list):
    signatures_list_out = list()
    for idx, label in enumerate(labels_list):
        if label in items_list:
            signatures_list_out.append(signatures_list[idx])
    return signatures_list_out


class DataLoader:
    def __init__(self):
        self._signatures = []
        self._labels = []

        self.group_labels = None
        self.cameras_labels = None

    def load_datasets(
        self,
        group_id: int,
        imagings_ids: Optional[list[int]] = None,
        cameras_labels: Optional[list[str]] = None,
    ):
        self.group_labels = get_labels_by_group(group_id)

        if imagings_ids is None:
            imagings_ids = list(settings.dataset_id_map.keys())
        if cameras_labels is None:
            self.cameras_labels = settings.cameras_id_labels
        else:
            self.cameras_labels = cameras_labels

        _ = [self._load_dataset(id) for id in imagings_ids]

    def _load_dataset(self, imagings_ids: int):
        dataset_name = settings.dataset_id_map[imagings_ids]
        dataset_dir = settings.data_dir / dataset_name

        file_signatures_vnir = dataset_dir / settings.file_signatures_vnir
        file_signatures_swir = dataset_dir / settings.file_signatures_swir
        file_labels = dataset_dir / settings.file_labels

        signatures_vnir = pd.read_csv(file_signatures_vnir, header=None)
        signatures_swir = pd.read_csv(file_signatures_swir, header=None)
        labels = pd.read_csv(file_labels, header=None)

        pass
