import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder


def relevant_features(relevances: np.ndarray, bands: np.ndarray) -> Figure:
    y = relevances.mean(axis=0)
    # remove the pre-defined noisy bands and assign values to the remaining places
    # y[np.delete(np.arange(len(BANDS_ORIGINAL)), NOISY_BANDS)] = relevances
    # first append pre-defined noisy indices at the end (least relevant features)
    indices_by_relevance = np.argsort(y)[::-1]
    max_features = len(indices_by_relevance)

    features_rang = np.full(len(indices_by_relevance), 0)
    for idx, feature in enumerate(indices_by_relevance):
        features_rang[feature] = max_features - idx

    gradient = np.vstack((features_rang, features_rang))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    # ax.set_title("Relavant features", fontsize=14)

    w_start, w_stop = bands[0], bands[-1]

    cmap = cm.get_cmap("viridis", 8)
    cmap.set_under("white")
    ax.imshow(
        gradient, aspect="auto", cmap=cmap, extent=[w_start, w_stop, 1, 0], vmin=1
    )

    ax.set_xlabel("Wavelength [nm]")
    ax.xaxis.label.set_size(12)
    ax.set_yticklabels([])
    return fig


def relevant_amplitudes(relevances: np.ndarray, bands: np.ndarray) -> Figure:
    y = relevances.mean(axis=0)
    x = bands

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    plt.plot(x, y, c="gray", linewidth=0.3, zorder=1)
    plt.scatter(x, y, c=y, s=3, cmap="viridis", zorder=2)

    ax.set_xlim(x.min(), x.max())
    # ax.set_ylim(0, 1.2)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Relevance")

    plt.tick_params(labelsize=22)
    ax.set_xlabel("Wavelength [nm]", fontsize=24)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)

    # peak_heights_dict, peak_indexes = find_signal_peaks(y)
    # [plt.text(x[idx], peak_heights_dict[idx], int(x[idx]), fontsize=18) for idx in peak_indexes]
    # plt.axhline(y=1, color="r", linestyle="--", alpha=0.6)
    return fig


def confusion_matrix_display(
    model: BaseEstimator,
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
) -> Figure:
    y_encoded = np.array(encoder.fit_transform(y))
    classes = encoder.classes_

    y_true_ = []
    y_pred_ = []

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    for train_index, test_index in rskf.split(X, y, groups=None):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        model = clone(model)
        model.fit(x_train, y_train)  # type: ignore
        y_pred = model.predict(x_test)  # type: ignore

        y_true_.extend(y_test)
        y_pred_.extend(y_pred)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true_, y_pred_, display_labels=classes, normalize="true"
    )
    cm_display.plot()
    return fig


def signatures_display(
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str = "Spectral bands",
    y_label: str = "",
) -> Figure:
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
    return fig


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
