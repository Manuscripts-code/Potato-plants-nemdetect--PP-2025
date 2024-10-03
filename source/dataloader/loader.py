import numpy as np
import pandas as pd

from source.core import logger, settings

from .helpers import count_unique_labels, get_labels_by_group

FILE_LABELS: str = "labels.csv"
FILE_SIGNATURES_VNIR: str = "signatures_vnir.csv"
FILE_SIGNATURES_SWIR: str = "signatures_swir.csv"

DATASET_ID_MAP: dict = {1: "imaging1", 2: "imaging2", 3: "imaging3"}


class DataLoader:
    def __init__(self):
        self._signatures = []
        self._labels = []

        self.group_labels_map = None
        self.cameras_labels = None

    def load_datasets(
        self,
        group_id: int,
        imagings_ids: list[int],
        cameras_labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imagings_ids.sort()
        logger.info(f"Using group id: {group_id}")
        logger.info(f"Using imagings ids: {imagings_ids}")
        logger.info(f"Using camera labels: {cameras_labels}")

        self.group_labels_map = get_labels_by_group(group_id)
        self.cameras_labels = cameras_labels
        logger.info(f"Labels mapping used: {self.group_labels_map}")

        _ = [self._load_dataset(id) for id in imagings_ids]

        signatures = pd.concat(self._signatures).to_numpy()
        labels = pd.concat(self._labels).to_numpy()
        # Currently meta has only imaging session index
        meta = np.array(
            [[idx] * len(item.to_list()) for idx, item in enumerate(self._labels)]
        ).flatten()
        logger.info(f"Label counts: {count_unique_labels(labels)}")
        return signatures, labels, meta

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
