import numpy as np
from scipy.signal import detrend, savgol_filter
from scipy.signal.windows import general_gaussian
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "SavgolWrapper",
    "FFTWrapper",
    "DerivWrapper",
    "MSCWrapper",
    "SNVTransformer",
    "DetrendTransformer",
]


class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=7, polyorder=2, deriv=2):
        self.win_length = win_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, self.polyorder, self.deriv]
        for signal in X:
            if self.win_length != 0:
                signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return np.array(signatures_sav)


class FFTWrapper(BaseEstimator, TransformerMixin):
    """
    https://nirpyresearch.com/fourier-spectral-smoothing-method/
    for derivatives Fourier derivative theorem used
    """

    def __init__(self, shape_param=1, sigma=10, deriv=2):
        self.shape_param = shape_param
        self.sigma = sigma
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_fft = []
        for signal in X:
            signal = self.FFT_filter(signal)
            signatures_fft.append(signal)
        return np.array(signatures_fft)

    def FFT_filter(self, signal):
        XX = np.hstack((signal, np.flip(signal)))
        win = np.roll(
            general_gaussian(XX.shape[0], self.shape_param, self.sigma),
            XX.shape[0] // 2,
        )
        fXX = np.fft.fft(XX)

        if self.deriv != 0:
            qq = (
                2
                * np.pi
                * np.arange(-XX.shape[0] // 2, XX.shape[0] // 2, 1)
                / XX.shape[0]
            )
            if self.deriv == 1:
                fXX = np.roll(
                    np.roll(fXX, -XX.shape[0] // 2) * (np.complex(0, 1) * qq),
                    XX.shape[0] // 2,
                )
            elif self.deriv == 2:
                fXX = np.roll(
                    np.roll(fXX, -XX.shape[0] // 2) * (-(qq**2)), XX.shape[0] // 2
                )

        return np.real(np.fft.ifft(fXX * win))[: signal.shape[0]]


class DerivWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, deriv=0):
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_diff = []
        for signal in X:
            signal = np.diff(signal, self.deriv)
            signatures_diff.append(signal)
        return np.array(signatures_diff)


class MSCWrapper(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction"""

    def fit(self, X, y=None):
        # mean centre correction
        for i in range(X.shape[0]):
            X[i, :] -= X[i, :].mean()

        # Get the reference spectrum. If not given, estimate it from the mean
        reference = None
        if reference is None:
            # Calculate mean
            self.ref = np.mean(X, axis=0)
        else:
            self.ref = reference
        return self

    def transform(self, X, y=None):
        # Define a new array and populate it with the corrected data
        data_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            # Run regression
            fit = np.polyfit(self.ref, X[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (X[i, :] - fit[0][1]) / fit[0][0]
        return data_msc


class SNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply SNV to each sample
        return np.array([(x - np.mean(x)) / np.std(x) for x in X])


class DetrendTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type="linear"):
        self.type = type  # 'linear' or 'constant'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply detrend to each sample
        return np.array([detrend(x, type=self.type) for x in X])
