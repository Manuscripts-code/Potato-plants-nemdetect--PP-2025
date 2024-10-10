import matplotlib.pyplot as plt
import numpy as np
import shap
import umap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


def shap_display(
    shap_values: np.ndarray,
    X: np.ndarray,
):
    shap.summary_plot(shap_values, X)
    plt.show()


def signatures_display(
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str = "Spectral bands",
    y_label: str = "",
):
    y_encoded = np.array(encoder.fit_transform(y))
    classes = encoder.classes_
    cmap = plt.get_cmap("Spectral")
    unique_labels = np.unique(y_encoded)
    no_colors = len(unique_labels)

    if no_colors > 2:
        colors = cmap(np.linspace(0, 1, no_colors))
    else:
        colors = ["darkgoldenrod", "forestgreen"]

    x_values = list(range(X.shape[1]))

    mean_values = [
        np.mean(X[y_encoded == unique_labels[idx]], axis=0) for idx in unique_labels
    ]
    std_values = [
        np.std(X[y_encoded == unique_labels[idx]], axis=0) for idx in unique_labels
    ]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    for idx in unique_labels:
        mean = mean_values[idx]
        std = std_values[idx]
        ax.plot(x_values, mean, color=colors[idx], label=classes[idx], alpha=0.6)
        ax.fill_between(
            x_values,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            color=colors[idx],
            alpha=0.2,
        )

    custom_lines = []
    for idx in unique_labels:
        custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel(x_label, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    # ax.set_ylim([0, 1])
    # ax.spines["bottom"].set_linewidth(2)
    # ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(data.columns, rotation=0)
    ax.legend(loc="upper right", fontsize=12, framealpha=1, frameon=False)

    plt.show()


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
