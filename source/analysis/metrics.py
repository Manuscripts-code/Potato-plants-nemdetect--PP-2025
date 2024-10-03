import numpy as np

# from siapy.models.metrics import calculate_classification_metrics
from sklearn.base import BaseEstimator

# from source.core import artifacts, logger
from sklearn.preprocessing import LabelEncoder


def calculate_metrics(
    model: BaseEstimator, encoder: LabelEncoder, X: np.ndarray, y: np.ndarray
):
    pass


# x_train, x_test, y_train, y_test = train_test_split(
#     X,
#     y_encoded,
#     test_size=0.2,
#     random_state=0,
#     shuffle=True,
#     stratify=y_encoded,
# )
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# return calculate_classification_metrics(y_test, y_pred), model, y_pred, y_test
