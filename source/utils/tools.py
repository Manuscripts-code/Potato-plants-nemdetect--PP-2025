import numpy as np


def smooth_signal(signal: np.ndarray, window_size: int = 10) -> np.ndarray:
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, kernel, mode="same")
    return smoothed_signal


def smooth_relevances(relevances: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(smooth_signal, axis=1, arr=np.abs(relevances)).mean(
        axis=0
    )
