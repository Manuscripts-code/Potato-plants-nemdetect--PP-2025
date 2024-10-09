import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


def shap_display(
    model: BaseEstimator,
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
):
    # y_encoded = np.array(encoder.fit_transform(y))
    pass


def umap_display(
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
) -> Figure:
    y_encoded = np.array(encoder.fit_transform(y))
    classes = encoder.classes_
    reducer = umap.UMAP(random_state=0)
    embeddings = reducer.fit_transform(X)

    y_encoded = y_encoded
    if meta is None:
        meta = np.zeros_like(y_encoded)

    cmap = plt.get_cmap("Spectral")
    if int(len(classes) / len(np.unique(meta))) > 2:
        _colors = cmap(np.linspace(0, 1, int(len(classes) / len(np.unique(meta)))))
    else:
        _colors = ["red", "blue"]
    _markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h"]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    for group_idx in np.unique(meta):
        embedding_masked = embeddings[meta == group_idx]
        y_data_encoded_masked = y_encoded[meta == group_idx]

        for class_idx in np.unique(y_data_encoded_masked):
            embedding_2masked = embedding_masked[y_data_encoded_masked == class_idx]

            classes_selected = np.array(classes)[np.unique(y_data_encoded_masked)]
            classes_selected_sorted = np.sort(classes_selected)
            classes_idx_map = {
                idx: np.where(classes_selected_sorted == item)[0][0]
                for idx, item in enumerate(classes_selected)
            }

            ax.scatter(
                embedding_2masked[:, 0],
                embedding_2masked[:, 1],
                s=50,
                color=_colors[
                    classes_idx_map[class_idx % len(np.unique(y_data_encoded_masked))]
                ],
                alpha=0.6,
                marker=_markers[group_idx],
                label=classes[class_idx] + f" s{group_idx}",
            )

        ax.legend(fontsize=18, framealpha=1)
        plt.setp(ax, xticks=[], yticks=[])

    return fig
