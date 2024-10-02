from typing import Optional

import numpy as np
import pandas as pd

from source.core import logger, settings

from .helpers import count_unique_labels, get_labels_by_group

FILE_LABELS: str = "labels.csv"
FILE_SIGNATURES_VNIR: str = "signatures_vnir.csv"
FILE_SIGNATURES_SWIR: str = "signatures_swir.csv"

DATASET_ID_MAP: dict = {1: "imaging1", 2: "imaging2", 3: "imaging3"}
CAMERAS_ID_LABELS: list = ["vnir", "swir"]


class DataLoader:
    def __init__(self):
        self._signatures = []
        self._labels = []

        self.group_labels_map = None
        self.cameras_labels = None

    def load_datasets(
        self,
        group_id: int,
        imagings_ids: Optional[list[int]] = None,
        cameras_labels: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Using group id: {group_id}")
        self.group_labels_map = get_labels_by_group(group_id)
        logger.info(f"Labels mapping used: {self.group_labels_map}")

        if imagings_ids is None:
            imagings_ids = list(DATASET_ID_MAP.keys())
        logger.info(f"Using imagings ids: {imagings_ids}")

        if cameras_labels is None:
            self.cameras_labels = CAMERAS_ID_LABELS
        else:
            self.cameras_labels = cameras_labels
        logger.info(f"Using camera labels: {self.cameras_labels}")

        _ = [self._load_dataset(id) for id in imagings_ids]

        signatures = pd.concat(self._signatures).to_numpy()
        labels = pd.concat(self._labels).to_numpy()
        logger.info(f"Label counts: {count_unique_labels(labels)}")
        return signatures, labels

    def _load_dataset(self, imaging_id: int):
        dataset_name = DATASET_ID_MAP[imaging_id]
        dataset_dir = settings.data_dir / dataset_name

        file_signatures_vnir = dataset_dir / FILE_SIGNATURES_VNIR
        file_signatures_swir = dataset_dir / FILE_SIGNATURES_SWIR
        file_labels = dataset_dir / FILE_LABELS

        signatures_vnir = pd.read_csv(file_signatures_vnir, header=None)
        signatures_swir = pd.read_csv(file_signatures_swir, header=None)
        labels = pd.read_csv(file_labels, header=None)

        if (
            self.cameras_labels is not None
            and "vnir" in self.cameras_labels
            and "swir" not in self.cameras_labels
        ):
            logger.info("Only data from vnir camera used.")
            signatures = signatures_vnir
        elif (
            self.cameras_labels is not None
            and "swir" in self.cameras_labels
            and "vnir" not in self.cameras_labels
        ):
            logger.info("Only data from swir camera used.")
            signatures = signatures_swir
        else:
            logger.info("Data from both cameras (vnir and swir) used.")
            signatures = pd.concat([signatures_vnir, signatures_swir], axis=1)

        reverse_mapping = {
            number: group_label
            for group_label, numbers in self.group_labels_map.items()  # type: ignore
            for number in numbers
        }
        labels = labels[0].map(reverse_mapping)

        # Remove NaN values from both signatures and labels
        valid_indices = labels.dropna().index
        signatures = signatures.loc[valid_indices]
        labels = labels.dropna()

        self._signatures.append(signatures)
        self._labels.append(labels)
